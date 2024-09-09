from torchtune.data import InstructTemplate
from typing import Any, Dict, Mapping, Optional


class SummarizeTemplate(InstructTemplate):
    """
    Prompt template to format datasets for summarization tasks.

    .. code-block:: text

        Summarize this dialogue:
        <YOUR DIALOGUE HERE>
        ---
        Summary:

    """

    template = "Summarize this dialogue in a single sentence:\n{dialogue}\n---\nSummary:\n"

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from dialogue.

        Args:
            sample (Mapping[str, Any]): a single data sample with dialog
            column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names
                in the template to the column names in the sample. If None, assume these are identical.

        Examples:
            >>> # Simple dialogue
            >>> SummarizeTemplate.format(sample={"dialogue": "Hello, how are you? Did you know the capital of France is Paris?"})
            Summarize this dialogue:
            Hello, how are you? Did you know the capital of France is Paris?
            ---
            Summary:

            >>> # Dialogue with column map where the 'dialogue' key is actually named 'prompt' in the given sample
            >>> SummarizeTemplate.format(
            ...     sample={"prompt": "Hello, how are you? Did you know the capital of France is Paris?"},
            ...     column_map={"dialogue": "prompt"}
            ... )
            Summarize this dialogue:
            Hello, how are you? Did you know the capital of France is Paris?
            ---
            Summary:

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_dialogue = column_map.get("dialogue", "dialogue")

        prompt = cls.template.format(dialogue=sample[key_dialogue])
        return prompt