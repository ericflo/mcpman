import json
import unittest
from unittest.mock import patch, MagicMock
from src.mcpman.llm_clients.anthropic import AnthropicClient


class TestClaudeDuplicateToolCalls(unittest.TestCase):
    def setUp(self):
        # Create a client with mock credentials
        self.client = AnthropicClient(
            api_key="test_key",
            api_url="https://api.anthropic.com/v1/messages",
            model_name="claude-3-7-sonnet-20250219"
        )

    def test_normalize_response_handles_duplicate_tool_calls(self):
        """Test that the response normalization deduplicates tool calls to the same function."""
        # Mock Claude API response with duplicate tool calls
        claude_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me calculate the square root twice."},
                {"type": "tool_use", "id": "call_1", "name": "calculator_sqrt", "input": {"number": 4}},
                {"type": "text", "text": "Let me try that again."},
                {"type": "tool_use", "id": "call_2", "name": "calculator_sqrt", "input": {"number": 4}},
                {"type": "tool_use", "id": "call_3", "name": "calculator_add", "input": {"a": 2, "b": 2}}
            ]
        }

        # Call the normalization function
        normalized = self.client._normalize_claude_response(claude_response)

        # Assert that the response contains only two tool calls (one sqrt, one add)
        self.assertEqual(len(normalized["tool_calls"]), 2)
        
        # Check that the tool calls are the expected ones
        tool_names = [tc["function"]["name"] for tc in normalized["tool_calls"]]
        self.assertIn("calculator_sqrt", tool_names)
        self.assertIn("calculator_add", tool_names)
        self.assertEqual(tool_names.count("calculator_sqrt"), 1)

    @patch('httpx.Client')
    def test_get_response_deduplicates_tool_calls(self, mock_client):
        """Test the full get_response method with a response containing duplicate tool calls."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me calculate square roots."},
                {"type": "tool_use", "id": "call_1", "name": "calculator_sqrt", "input": {"number": 4}},
                {"type": "tool_use", "id": "call_2", "name": "calculator_sqrt", "input": {"number": 9}},
                {"type": "tool_use", "id": "call_3", "name": "calculator_sqrt", "input": {"number": 16}}
            ]
        }
        
        # Configure the mock client
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        # Don't mock json.dumps to avoid recursion issues
        
        # Call the method
        result = self.client.get_response([{"role": "user", "content": "Calculate some square roots"}])
        
        # Verify that there's only one sqrt tool call in the result
        self.assertIn("tool_calls", result)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "calculator_sqrt")


if __name__ == "__main__":
    unittest.main()