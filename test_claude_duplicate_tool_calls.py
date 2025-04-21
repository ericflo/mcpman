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

    def test_filters_exact_duplicate_tool_calls(self):
        """Test that the response normalization deduplicates exact tool calls (same name and args)."""
        # Mock Claude API response with exact duplicate tool calls
        claude_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me calculate the square root twice."},
                {"type": "tool_use", "id": "call_1", "name": "calculator_sqrt", "input": {"number": 4}},
                {"type": "text", "text": "Let me try that again."},
                # Exact duplicate (same name, same args)
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
        
    def test_allows_same_tool_different_args(self):
        """Test that same tool with different arguments is allowed."""
        # Mock Claude API response with same tool but different arguments
        claude_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me calculate multiple square roots."},
                {"type": "tool_use", "id": "call_1", "name": "calculator_sqrt", "input": {"number": 4}},
                {"type": "tool_use", "id": "call_2", "name": "calculator_sqrt", "input": {"number": 9}},
                {"type": "tool_use", "id": "call_3", "name": "calculator_sqrt", "input": {"number": 16}},
                {"type": "tool_use", "id": "call_4", "name": "calculator_add", "input": {"a": 2, "b": 2}}
            ]
        }

        # Call the normalization function
        normalized = self.client._normalize_claude_response(claude_response)

        # Should have 4 tool calls (3 unique sqrt calls + 1 add call)
        self.assertEqual(len(normalized["tool_calls"]), 4)
        
        # Check that all tool calls went through
        tool_names = [tc["function"]["name"] for tc in normalized["tool_calls"]]
        self.assertEqual(tool_names.count("calculator_sqrt"), 3)
        self.assertEqual(tool_names.count("calculator_add"), 1)
        
        # Verify the arguments are preserved for each sqrt call
        sqrt_args = []
        for tc in normalized["tool_calls"]:
            if tc["function"]["name"] == "calculator_sqrt":
                args = json.loads(tc["function"]["arguments"])
                sqrt_args.append(args["number"])
                
        self.assertIn(4, sqrt_args)
        self.assertIn(9, sqrt_args)
        self.assertIn(16, sqrt_args)
        
    def test_handles_args_with_different_order(self):
        """Test that arguments with same values but different key order are considered equal."""
        # Mock Claude API response with same args in different orders
        claude_response = {
            "id": "msg_123", 
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me test with different arg orders."},
                {"type": "tool_use", "id": "call_1", "name": "calculator_func", 
                 "input": {"x": 1, "y": 2, "z": 3}},
                # Same args with different key order
                {"type": "tool_use", "id": "call_2", "name": "calculator_func", 
                 "input": {"y": 2, "z": 3, "x": 1}}
            ]
        }
        
        # Call the normalization function
        normalized = self.client._normalize_claude_response(claude_response)
        
        # Should only have 1 tool call since the arguments are equivalent
        self.assertEqual(len(normalized["tool_calls"]), 1)

    @patch('httpx.Client')
    def test_get_response_deduplicates_tool_calls(self, mock_client):
        """Test the full get_response method with a response containing duplicate tool calls."""
        # Set up mock response with IDENTICAL duplicate calls to calculator_sqrt
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me calculate square roots."},
                {"type": "tool_use", "id": "call_1", "name": "calculator_sqrt", "input": {"number": 4}},
                # Duplicate with same args
                {"type": "tool_use", "id": "call_2", "name": "calculator_sqrt", "input": {"number": 4}},
                # Different args - should be kept
                {"type": "tool_use", "id": "call_3", "name": "calculator_sqrt", "input": {"number": 16}}
            ]
        }
        
        # Configure the mock client
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        # Call the method
        result = self.client.get_response([{"role": "user", "content": "Calculate some square roots"}])
        
        # Verify that we have 2 sqrt tool calls (one with number=4, one with number=16)
        # The duplicate call to sqrt(4) should be filtered out
        self.assertIn("tool_calls", result)
        self.assertEqual(len(result["tool_calls"]), 2)
        
        # Extract arguments to verify the right calls were kept
        sqrt_args = []
        for tc in result["tool_calls"]:
            self.assertEqual(tc["function"]["name"], "calculator_sqrt")
            args = json.loads(tc["function"]["arguments"])
            sqrt_args.append(args["number"])
            
        # Should have 4 and 16, with 4 appearing only once
        self.assertIn(4, sqrt_args)
        self.assertIn(16, sqrt_args)
        self.assertEqual(sqrt_args.count(4), 1)


if __name__ == "__main__":
    unittest.main()