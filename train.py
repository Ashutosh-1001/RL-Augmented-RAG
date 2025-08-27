from ragrl import ReinforcedRAG, load_text_lines


def train_model(rag_system, queries, answers, epochs=10):
    for epoch in range(epochs):
        total_loss = 0.0
        total_reward = 0.0
        valid_steps = 0
        for query, answer in zip(queries, answers):
            loss, reward = rag_system.train_on_query(query, answer)
            if loss != 0 or reward != 0:
                total_loss += loss
                total_reward += reward
                valid_steps += 1
        if valid_steps > 0:
            avg_loss = total_loss / valid_steps
            avg_reward = total_reward / valid_steps
            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")
        else:
            print(f"Epoch {epoch + 1}: No valid training data")


if __name__ == "__main__":
    rag = ReinforcedRAG("sample.txt")
    queries = load_text_lines("queries.txt")
    answers = load_text_lines("answers.txt")
    train_model(rag, queries, answers, epochs=10)
