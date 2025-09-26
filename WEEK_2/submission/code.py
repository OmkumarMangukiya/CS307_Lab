import re
import heapq
import nltk
import sys

class AStarPlagiarismDetector:
    def __init__(self, plagiarism_threshold=0.8):
        self.threshold = plagiarism_threshold
        self._download_nltk_punkt()

    def _download_nltk_punkt(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("NLTK 'punkt' tokenizer not found. Downloading...")
            nltk.download('punkt')
            print("Download complete.")

    def preprocess(self, text):
        sentences = nltk.sent_tokenize(text)
        normalized_sentences = []
        for sentence in sentences:
            s = sentence.lower()
            s = re.sub(r'[^\w\s]', '', s)
            normalized_sentences.append(s.strip())
        return [s for s in normalized_sentences if s]

    def edit_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def distance_heuristic(self, state, goal_state):
        """
        Calculate heuristic distance to goal state using Manhattan distance.
        This estimates the minimum number of skip operations needed.
        """
        i, j = state
        goal_i, goal_j = goal_state
        return abs(goal_i - i) + abs(goal_j - j)

    def a_star_align(self, doc1_sents, doc2_sents):
        start_state = (0, 0)
        goal_state = (len(doc1_sents), len(doc2_sents))

        doc1_lens = [len(s) for s in doc1_sents]
        doc2_lens = [len(s) for s in doc2_sents]

        # The priority queue now stores (f_cost, g_cost, state, path)
        open_list = [(0, 0, start_state, [])]
        visited_states = {}

        while open_list:
            f_cost, g_cost, current_state, path = heapq.heappop(open_list)

            if current_state in visited_states and visited_states[current_state] <= g_cost:
                continue
            visited_states[current_state] = g_cost

            if current_state == goal_state:
                return path, g_cost

            i, j = current_state

            # Successor 1: Align sentences
            if i < len(doc1_sents) and j < len(doc2_sents):
                cost = self.edit_distance(doc1_sents[i], doc2_sents[j])
                new_g = g_cost + cost
                successor_state = (i + 1, j + 1)
                
                # --- HEURISTIC CALCULATION ---
                h = self.distance_heuristic(successor_state, goal_state)
                new_f = new_g + h
                
                new_path = path + [('align', i, j, cost)]
                heapq.heappush(open_list, (new_f, new_g, successor_state, new_path))

            # Successor 2: Skip sentence in document 1
            if i < len(doc1_sents):
                cost = doc1_lens[i]
                new_g = g_cost + cost
                successor_state = (i + 1, j)

                # --- HEURISTIC CALCULATION ---
                h = self.distance_heuristic(successor_state, goal_state)
                new_f = new_g + h
                
                new_path = path + [('skip_doc1', i, -1, cost)]
                heapq.heappush(open_list, (new_f, new_g, successor_state, new_path))

            # Successor 3: Skip sentence in document 2
            if j < len(doc2_sents):
                cost = doc2_lens[j]
                new_g = g_cost + cost
                successor_state = (i, j + 1)
                
                # --- HEURISTIC CALCULATION ---
                h = self.distance_heuristic(successor_state, goal_state)
                new_f = new_g + h
                
                new_path = path + [('skip_doc2', -1, j, cost)]
                heapq.heappush(open_list, (new_f, new_g, successor_state, new_path))

        return [], float('inf')

    def report_results(self, path, doc1_sents, doc2_sents, total_cost):
        # This function remains the same
        print("\n--- Plagiarism Detection Report ---")
        print(f"Optimal Alignment Cost (Total Edit Distance): {total_cost}")
        plagiarised_count = 0
        
        for operation, i, j, cost in path:
            if operation == 'align':
                s1 = doc1_sents[i]
                s2 = doc2_sents[j]
                max_len = max(len(s1), len(s2))
                similarity = 1 - (cost / max_len) if max_len > 0 else 1.0
                
                print(f"\n[ALIGN] Doc 1 Sent {i+1} <-> Doc 2 Sent {j+1}")
                print(f"  Doc 1: '{s1}'")
                print(f"  Doc 2: '{s2}'")
                print(f"  Similarity: {similarity:.2f} (Cost: {cost})")
                
                if similarity >= self.threshold:
                    print("  -> STATUS: POTENTIAL PLAGIARISM DETECTED")
                    plagiarised_count += 1
                else:
                    print("  -> STATUS: OK")
                    
            elif operation == 'skip_doc1':
                print(f"\n[SKIP] Doc 1 Sent {i+1}: '{doc1_sents[i]}' (Cost: {cost})")
            elif operation == 'skip_doc2':
                print(f"\n[SKIP] Doc 2 Sent {j+1}: '{doc2_sents[j]}' (Cost: {cost})")
        
        print("\n--- Summary ---")
        if plagiarised_count > 0:
            print(f"Found {plagiarised_count} sentence pair(s) exceeding the {self.threshold:.0%} similarity threshold.")
        else:
            print(f"No sentence pairs met the {self.threshold:.0%} similarity threshold for plagiarism.")

    def detect(self, doc1_text, doc2_text):
        # This function remains the same
        print(f"Document 1: \"{doc1_text}\"")
        print(f"Document 2: \"{doc2_text}\"\n")
        
        doc1_sents = self.preprocess(doc1_text)
        doc2_sents = self.preprocess(doc2_text)
        
        print(f"Doc 1 sentences: {len(doc1_sents)}, Doc 2 sentences: {len(doc2_sents)}")

        if not doc1_sents or not doc2_sents:
            print("One or both documents are empty after preprocessing. Cannot compare.")
            return

        alignment_path, total_cost = self.a_star_align(doc1_sents, doc2_sents)
        
        self.report_results(alignment_path, doc1_sents, doc2_sents, total_cost)


if __name__ == '__main__':
    # This part remains the same
    if len(sys.argv) != 3:
        print("Usage: python plagiarism_detector.py <file1.txt> <file2.txt>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            doc1_text = f.read()
        with open(file2_path, 'r', encoding='utf-8') as f:
            doc2_text = f.read()
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    detector = AStarPlagiarismDetector(plagiarism_threshold=0.8)
    
    print("=" * 60)
    print(f"Comparing Document 1 ('{file1_path}') and Document 2 ('{file2_path}')")
    print("=" * 60)
    
    detector.detect(doc1_text, doc2_text)