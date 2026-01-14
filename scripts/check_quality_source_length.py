"""Quick script to estimate average source_text length in QuALITY dataset."""
from solib.data.loading import QuALITY

def main():
    # Load sample of 100 questions
    print("Loading QuALITY dataset (sample of 100 questions)...")
    dataset = QuALITY.data(limit=100)
    
    # Calculate source_text lengths
    lengths = []
    questions_with_source = 0
    questions_without_source = 0
    
    for question in dataset:
        if question.source_text:
            length = len(question.source_text)
            lengths.append(length)
            questions_with_source += 1
        else:
            questions_without_source += 1
    
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        median_length = sorted(lengths)[len(lengths) // 2]
        
        print(f"\nResults from {len(dataset)} questions:")
        print(f"  Questions with source_text: {questions_with_source}")
        print(f"  Questions without source_text: {questions_without_source}")
        print(f"\nSource text length statistics (characters):")
        print(f"  Average: {avg_length:.1f}")
        print(f"  Median:  {median_length}")
        print(f"  Min:     {min_length}")
        print(f"  Max:     {max_length}")
        
        # Also show word counts (approximate)
        word_counts = [len(source_text.split()) for source_text in [q.source_text for q in dataset if q.source_text]]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        print(f"\nWord count statistics:")
        print(f"  Average words: {avg_words:.1f}")
    else:
        print("No questions with source_text found!")

if __name__ == "__main__":
    main()
