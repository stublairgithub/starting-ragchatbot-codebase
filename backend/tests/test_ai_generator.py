import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock anthropic module before importing AIGenerator
sys.modules['anthropic'] = MagicMock()

from ai_generator import AIGenerator


class TestSequentialToolCalling:
    """Tests for sequential tool calling behavior"""

    @pytest.fixture
    def mock_tool_manager(self):
        """Mock tool manager that returns predictable results"""
        manager = Mock()
        manager.execute_tool.return_value = "Mock tool result"
        return manager

    def _create_tool_use_response(self, tool_name, tool_input, tool_id="test-id"):
        """Helper to create mock tool_use response"""
        response = Mock()
        response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = tool_name
        tool_block.input = tool_input
        tool_block.id = tool_id
        response.content = [tool_block]
        return response

    def _create_text_response(self, text):
        """Helper to create mock text response"""
        response = Mock()
        response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.type = "text"
        text_block.text = text
        response.content = [text_block]
        return response

    @patch('ai_generator.anthropic.Anthropic')
    def test_direct_answer_no_tools(self, mock_anthropic):
        """Claude answers directly - verify 1 API call, no tools executed"""
        # Setup: Claude returns text immediately
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = self._create_text_response("Direct answer")

        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response("Hello", tools=[], tool_manager=None)

        assert result == "Direct answer"
        assert mock_client.messages.create.call_count == 1

    @patch('ai_generator.anthropic.Anthropic')
    def test_single_tool_call(self, mock_anthropic, mock_tool_manager):
        """Claude uses one tool then answers - verify 2 API calls"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [
            self._create_tool_use_response("search_course_content", {"query": "test"}),
            self._create_text_response("Answer after search")
        ]

        generator = AIGenerator("test-key", "test-model")
        tools = [{"name": "search_course_content", "description": "Search"}]
        result = generator.generate_response("Search for X", tools=tools, tool_manager=mock_tool_manager)

        assert result == "Answer after search"
        assert mock_client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once()

    @patch('ai_generator.anthropic.Anthropic')
    def test_two_sequential_tool_calls(self, mock_anthropic, mock_tool_manager):
        """Claude uses tools twice - verify 3 API calls, 2 tool executions"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [
            self._create_tool_use_response("get_course_outline", {"course_name": "MCP"}, "id-1"),
            self._create_tool_use_response("search_course_content", {"query": "topic"}, "id-2"),
            self._create_text_response("Final answer")
        ]

        generator = AIGenerator("test-key", "test-model")
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        result = generator.generate_response("Complex query", tools=tools, tool_manager=mock_tool_manager)

        assert result == "Final answer"
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_max_rounds_enforced(self, mock_anthropic, mock_tool_manager):
        """After 2 tool rounds, final call made without tools"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        # Claude keeps requesting tools
        mock_client.messages.create.side_effect = [
            self._create_tool_use_response("search", {"q": "1"}, "id-1"),
            self._create_tool_use_response("search", {"q": "2"}, "id-2"),
            self._create_text_response("Forced final answer")  # 3rd call has no tools
        ]

        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response("Query", tools=[{"name": "search"}], tool_manager=mock_tool_manager)

        assert result == "Forced final answer"
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify 3rd API call was made WITHOUT tools parameter
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_error_passed_to_claude(self, mock_anthropic, mock_tool_manager):
        """Tool execution error is captured and passed to Claude"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [
            self._create_tool_use_response("search", {"q": "test"}),
            self._create_text_response("Handled the error")
        ]
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response("Query", tools=[{"name": "search"}], tool_manager=mock_tool_manager)

        # Should not raise, error passed to Claude
        assert result == "Handled the error"

    @patch('ai_generator.anthropic.Anthropic')
    def test_early_termination_when_no_more_tools_needed(self, mock_anthropic, mock_tool_manager):
        """Claude stops requesting tools before max rounds - verify early termination"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        # First round: tool use, second round: direct answer (no more tools)
        mock_client.messages.create.side_effect = [
            self._create_tool_use_response("search", {"q": "test"}, "id-1"),
            self._create_text_response("Found what I needed")
        ]

        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response("Query", tools=[{"name": "search"}], tool_manager=mock_tool_manager)

        assert result == "Found what I needed"
        # Only 2 API calls (initial + after first tool), not 3
        assert mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
