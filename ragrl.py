import os
import torch
import torch.nn as nn
import torch.optim as optim
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        return self.layers(x)


class ReinforcedRAG:
    def __init__(self, data_path):
        self.load_text_data(data_path)
        self.setup_embedding_model()
        self.setup_vector_retriever()
        self.setup_policy_network()
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        self.minimum_docs = 1

    def load_text_data(self, filepath):
        loader = TextLoader(filepath)
        documents = loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separator="\n",
            length_function=len
        )
        self.document_chunks = splitter.split_documents(documents)
        print(f"Loaded {len(self.document_chunks)} document chunks from {filepath}")

    def setup_embedding_model(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def setup_vector_retriever(self):
        self.vector_store = FAISS.from_documents(self.document_chunks, self.embedding_model)
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )

    def setup_policy_network(self):
        embedding_dim = 384  
        state_dim = embedding_dim * 2 
        self.policy_net = PolicyNetwork(state_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

    def get_state_vector(self, query, document):
        try:
            query_embedding = self.embedding_model.embed_query(query)
            doc_embedding = self.embedding_model.embed_query(document.page_content)
            combined_embedding = query_embedding + doc_embedding
            return torch.FloatTensor(combined_embedding)
        except Exception:
            return None

    def compute_reward(self, query, documents, generated_answer):
        try:
            answer_text = generated_answer.lower()
            query_words = set(query.lower().split())
            answer_words = set(answer_text.split())
            word_overlap = len(query_words.intersection(answer_words)) / max(len(query_words), 1)
            unique_docs = len({d.page_content[:50] for d in documents})
            diversity_score = unique_docs / len(documents)
            reward = 0.7 * word_overlap + 0.3 * diversity_score
            return reward
        except Exception:
            return 0.0

    def train_on_query(self, query, correct_answer):
        try:
            candidate_documents = self.retriever.get_relevant_documents(query)
            if len(candidate_documents) < self.minimum_docs or len(candidate_documents) == 1:
                return 0.0, 0.0  # Skip if not enough documents to rank

            states = [self.get_state_vector(query, doc) for doc in candidate_documents]
            valid_states = [s for s in states if s is not None]
            if len(valid_states) < 2:
                return 0.0, 0.0  

            scores = torch.stack([self.policy_net(state) for state in valid_states]).squeeze(-1)
            probabilities = torch.softmax(scores, dim=0)

            num_samples = min(len(valid_states), probabilities.size(0))
            selected_indices = torch.multinomial(probabilities, num_samples, replacement=False)
            ranked_docs = [candidate_documents[i] for i in selected_indices]

            context_text = "\n".join([doc.page_content for doc in ranked_docs[:3]])
            answer = self.llm([
                SystemMessage(content=f"Answer based on:\n{context_text}"),
                HumanMessage(content=query)
            ]).content

            reward = self.compute_reward(query, ranked_docs, answer)
            log_probs = torch.log(probabilities.gather(0, selected_indices))
            loss = -torch.mean(log_probs) * reward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item(), reward

        except Exception as e:
            print(f"Error during training step: {str(e)}")
            return 0.0, 0.0


def load_text_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def train_model(rag_system, queries, answers, epochs=10):
    for epoch in range(epochs):
        total_loss, total_reward, valid_steps = 0.0, 0.0, 0
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
    rag_system = ReinforcedRAG("sample.txt")

    queries = load_text_lines("queries.txt")
    answers = load_text_lines("answers.txt")

    train_model(rag_system, queries, answers, epochs=10)
