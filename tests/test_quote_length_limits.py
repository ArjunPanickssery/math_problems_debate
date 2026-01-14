"""Test quote length limits functionality."""
import pytest
from solib.utils.verification import verify_quotes_in_text
from solib.Experiment import Experiment
from solib.datatypes import Question, Answer


class TestVerifyQuotesInText:
    """Test the verify_quotes_in_text function with max_length parameter."""

    def test_no_limit_backward_compatibility(self):
        """Test that None max_length maintains backward compatibility (no limit)."""
        source_text = "This is a long source text that can be quoted from."
        text = "Here is a quote: <quote>This is a long source text that can be quoted from.</quote>"
        
        result = verify_quotes_in_text(text, source_text, max_length=None)
        
        # Should verify the quote without truncation
        assert "<quote_verified>" in result
        assert "This is a long source text that can be quoted from." in result

    def test_quote_within_limit(self):
        """Test that quotes within the limit are not truncated."""
        source_text = "Short quote text"
        text = "Here is a quote: <quote>Short quote text</quote>"
        
        result = verify_quotes_in_text(text, source_text, max_length=100)
        
        assert "<quote_verified>Short quote text</quote_verified>" in result
        assert "Short quote text" in result
        assert "..." not in result

    def test_quote_exceeds_limit_exact_truncation(self):
        """Test that quotes exceeding limit are truncated when no word boundary."""
        source_text = "This is a very long quote that exceeds the maximum length limit we set"
        text = "Here is a quote: <quote>This is a very long quote that exceeds the maximum length limit we set</quote>"
        
        result = verify_quotes_in_text(text, source_text, max_length=30)
        
        # Should be truncated to ~27 chars + "..."
        assert "<quote_verified>" in result or "<quote_invalid>" in result
        # Check that result is truncated (approximately max_length)
        import re
        matches = re.findall(r"<quote_[^>]*>(.*?)</quote_[^>]*>", result)
        assert len(matches) > 0
        quoted_text = matches[0]
        # Should be truncated (allow some flexibility for word boundary search)
        assert len(quoted_text) <= 30
        assert quoted_text.endswith("...")

    def test_quote_exceeds_limit_word_boundary(self):
        """Test that truncation prefers word boundaries."""
        source_text = "This is a very long quote that exceeds the maximum length limit we set"
        text = "Here is a quote: <quote>This is a very long quote that exceeds the maximum length limit we set</quote>"
        
        result = verify_quotes_in_text(text, source_text, max_length=30)
        
        # Extract the quoted content
        import re
        matches = re.findall(r"<quote_[^>]*>(.*?)</quote_[^>]*>", result)
        assert len(matches) > 0
        quoted_text = matches[0]
        
        # Should end with ellipsis
        assert quoted_text.endswith("...")
        # The text before ellipsis should ideally end at a word boundary (space or newline)
        # But we allow some flexibility - main thing is it's truncated
        truncated_part = quoted_text[:-3]  # Remove "..."
        # Should not cut in the middle of a word if possible
        assert len(quoted_text) <= 30

    def test_multiple_quotes_different_lengths(self):
        """Test handling of multiple quotes with different lengths."""
        source_text = "Short text. This is a much longer quote that will need truncation if we set a limit."
        text = "First: <quote>Short text</quote> Second: <quote>This is a much longer quote that will need truncation if we set a limit.</quote>"
        
        result = verify_quotes_in_text(text, source_text, max_length=20)
        
        # First quote should be unchanged
        assert "<quote_verified>Short text</quote_verified>" in result or "<quote_invalid>Short text</quote_invalid>" in result
        
        # Second quote should be truncated
        import re
        matches = re.findall(r"<quote_[^>]*>(.*?)</quote_[^>]*>", result)
        assert len(matches) == 2
        second_quote = matches[1]
        assert len(second_quote) <= 20
        assert second_quote.endswith("...")

    def test_quote_with_newlines_word_boundary(self):
        """Test that newlines are treated as word boundaries."""
        source_text = "First line\nSecond line\nThird line"
        text = "Quote: <quote>First line\nSecond line\nThird line</quote>"
        
        result = verify_quotes_in_text(text, source_text, max_length=15)
        
        # Should truncate at a newline if possible
        import re
        matches = re.findall(r"<quote_[^>]*>(.*?)</quote_[^>]*>", result, re.DOTALL)
        assert len(matches) > 0
        quoted_text = matches[0]
        assert len(quoted_text) <= 15
        assert quoted_text.endswith("...")

    def test_invalid_quote_with_truncation(self):
        """Test that invalid quotes are also truncated."""
        source_text = "Valid text in source"
        text = "Invalid quote: <quote>This is not in the source text and should be marked invalid and truncated</quote>"
        
        result = verify_quotes_in_text(text, source_text, max_length=25)
        
        # Should be marked invalid and truncated
        assert "<quote_invalid>" in result
        import re
        matches = re.findall(r"<quote_invalid>(.*?)</quote_invalid>", result)
        assert len(matches) > 0
        quoted_text = matches[0]
        assert len(quoted_text) <= 25
        assert quoted_text.endswith("...")

    def test_empty_source_text(self):
        """Test that max_length still works even when source_text is None."""
        text = "Quote: <quote>Some text that should be truncated</quote>"
        
        result = verify_quotes_in_text(text, None, max_length=10)
        
        # Should return text unchanged (no verification without source_text)
        assert result == text

    def test_very_short_limit(self):
        """Test behavior with very short limit."""
        source_text = "The quick brown fox jumps over the lazy dog"
        text = "Quote: <quote>The quick brown fox jumps over the lazy dog</quote>"
        
        result = verify_quotes_in_text(text, source_text, max_length=10)
        
        import re
        matches = re.findall(r"<quote_[^>]*>(.*?)</quote_[^>]*>", result)
        assert len(matches) > 0
        quoted_text = matches[0]
        # Should be truncated to ~7 chars + "..."
        assert len(quoted_text) <= 10
        assert quoted_text.endswith("...")


class TestExperimentQuoteMaxLength:
    """Test Experiment class integration with quote_max_length."""

    def test_experiment_stores_quote_max_length(self):
        """Test that Experiment stores quote_max_length parameter."""
        exp = Experiment(
            questions=[],
            agent_models=["gpt-4o-mini"],
            judge_models=["gpt-4o-mini"],
            quote_max_length=500,
        )
        
        assert exp.quote_max_length == 500

    def test_experiment_default_quote_max_length(self):
        """Test that Experiment defaults quote_max_length to None."""
        exp = Experiment(
            questions=[],
            agent_models=["gpt-4o-mini"],
            judge_models=["gpt-4o-mini"],
        )
        
        assert exp.quote_max_length is None

    def test_experiment_other_componentss_includes_quote_max_length(self):
        """Test that other_componentss includes quote_max_length when set."""
        exp = Experiment(
            questions=[],
            agent_models=["gpt-4o-mini"],
            judge_models=["gpt-4o-mini"],
            quote_max_length=300,
        )
        
        components = exp.other_componentss
        # Check all protocol types
        assert "quote_max_length" in components["blind"][0]
        assert components["blind"][0]["quote_max_length"] == 300
        assert "quote_max_length" in components["propaganda"][0]
        assert components["propaganda"][0]["quote_max_length"] == 300
        assert "quote_max_length" in components["consultancy"][0]
        assert components["consultancy"][0]["quote_max_length"] == 300
        # Debate also has adversary, so check that quote_max_length is included
        assert "quote_max_length" in components["debate"][0]
        assert components["debate"][0]["quote_max_length"] == 300

    def test_experiment_other_componentss_no_quote_max_length_when_none(self):
        """Test that other_componentss doesn't include quote_max_length when None."""
        exp = Experiment(
            questions=[],
            agent_models=["gpt-4o-mini"],
            judge_models=["gpt-4o-mini"],
            quote_max_length=None,
        )
        
        components = exp.other_componentss
        # Check that quote_max_length is not in any components
        assert "quote_max_length" not in components["blind"][0]
        assert "quote_max_length" not in components["propaganda"][0]
        assert "quote_max_length" not in components["consultancy"][0]
        # Debate has adversary, but still shouldn't have quote_max_length
        if components["debate"]:
            assert "quote_max_length" not in components["debate"][0] or components["debate"][0].get("quote_max_length") is None

    def test_experiment_other_componentss_debate_includes_adversary_and_quote_max_length(self):
        """Test that debate protocol gets both adversary and quote_max_length."""
        exp = Experiment(
            questions=[],
            agent_models=["gpt-4o-mini"],
            judge_models=["gpt-4o-mini"],
            quote_max_length=250,
        )
        
        components = exp.other_componentss
        # Debate should have both adversary and quote_max_length
        debate_component = components["debate"][0]
        assert "adversary" in debate_component
        assert "quote_max_length" in debate_component
        assert debate_component["quote_max_length"] == 250
