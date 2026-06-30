import streamlit as st
import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# SAFE RAGAS IMPORT
# ==========================================
RAGAS_AVAILABLE = False
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevance
    RAGAS_AVAILABLE = True
except ModuleNotFoundError as e:
    print(f"WARNING: RAGAS disabled - {e}")
except Exception:
    pass

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Enterprise RAG System", layout="wide", page_icon="🧠")
st.title("🧠 Enterprise RAG System")

# ==========================================
# SESSION STATE
# ==========================================
if "query" not in st.session_state:
    st.session_state.query = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

# ==========================================
# CUSTOM LLM WRAPPER FOR RAGAS
# ==========================================
class OllamaLLMWrapper(LLM):
    model_name: str = "llama3"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.get('message', {}).get('content', '')
    
    @property
    def _llm_type(self) -> str:
        return "ollama-custom"

# ==========================================
# 1. INITIALIZE COMPONENTS
# ==========================================
@st.cache_resource
def load_components():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return embedding_function, retriever

with st.status("Initializing System...", expanded=True) as status:
    embeddings, retriever = load_components()
    st.write("✅ Embeddings loaded")
    st.write("✅ Vector database connected")
    st.write("✅ RAGAS Enabled" if RAGAS_AVAILABLE else "⚠️ RAGAS disabled")
    status.update(label="System Ready!", state="complete", expanded=False)

# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Quick Questions")
    
    questions = {
        "🚗 Vehicle Production": "How many vehicles did Tesla produce in 2023?",
        "🏢 Move to Texas": "Why is Tesla moving its state of incorporation to Texas?",
        "⚠️ Risk Factors": "What are the main risk factors mentioned in the report?",
        "🏭 Factory Comparison": "Compare the manufacturing capacity and battery cell types at Gigafactory Texas versus Berlin-Brandenburg.",
    }
    
    for label, question in questions.items():
        if st.button(label, use_container_width=True, key=f"btn_{label}"):
            st.session_state.query = question
            st.session_state.button_clicked = True
    
    st.divider()
    
    custom_query = st.text_input(
        "Or type your question:",
        placeholder="e.g., Tell me about the factories",
        key="custom_query_input"
    )
    
    # Clean trailing commas/spaces and handle button vs text input conflict
    if custom_query and not st.session_state.button_clicked:
        st.session_state.query = custom_query.strip(" ,.")
        
    st.session_state.button_clicked = False
    
    st.divider()
    st.subheader("Retrieval Settings")
    k_value = st.slider("Chunks to retrieve (k)", 1, 10, 4) # Increased default to 4
    retriever.search_kwargs["k"] = k_value
    
    st.divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.query = ""
        st.rerun()

# ==========================================
# 3. MAIN APP LOGIC
# ==========================================
st.markdown("---")
user_query = st.session_state.query

if user_query:
    st.subheader("📝 Your Question")
    st.info(user_query)
    
    try:
        # RETRIEVE
        with st.spinner("🧠 Retrieving relevant documents..."):
            retrieved_docs = retriever.invoke(user_query)
            context_text = "\n\n".join([
    f"--- Document: {doc.metadata.get('source_filename', 'Unknown')} ---\n{doc.page_content}" 
    for doc in retrieved_docs
])
            
            # THE SUPER PROMPT (Fixes Texas Trap + Repetition)
            prompt = f"""You are a Tesla financial and technical analyst. 
Read the provided document excerpts and answer the question.
- If excerpts are from different years/files, compare them directly.
- Use bullet points.
- Do NOT comment on your process or repeat yourself.

DOCUMENT EXCERPTS:
{context_text}

QUESTION:
{user_query}

ANSWER:"""
        
        # STREAMING GENERATE
                # STREAMING GENERATE (FIXED VISUAL BUG)
        st.subheader("💬 AI Answer")
        
        # Create a single placeholder for the streaming text
        answer_placeholder = st.empty()
        
        response_stream = ollama.chat(
            model='llama3.1:8b',
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer strictly based on context. NEVER repeat yourself."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        full_answer = ""
        for chunk in response_stream:
            text_chunk = chunk.get('message', {}).get('content', '')
            full_answer += text_chunk
            # .markdown() on a st.empty() overwrites the previous line!
            answer_placeholder.markdown(full_answer + " ▌")
        
        # Final clean display without the cursor
        answer_placeholder.markdown(full_answer)
        
        
        # SAVE TO HISTORY
        st.session_state.history.append({
            "question": user_query,
            "answer": full_answer,
            "docs": retrieved_docs
        })
        
        # DISPLAY SOURCES (Shows exact file and page)
        st.subheader("📄 Source Chunks (Proof of Grounding)")
        cols = st.columns(3)
        
        for i, doc in enumerate(retrieved_docs):
            # Safely get filename and page
            source_file = doc.metadata.get("source_filename", "Unknown File")
            page_num = doc.metadata.get("page", "N/A")
            
            with cols[i % 3]:
                with st.expander(f"📄 {source_file} (Pg {page_num})", expanded=(i < 3)):
                    st.text(doc.page_content[:800] + "...")
        
        # RAGAS EVALUATION
        if RAGAS_AVAILABLE:
            st.markdown("---")
            st.subheader("🔍 RAGAS Evaluation")
            
            with st.spinner("Running RAGAS evaluation..."):
                try:
                    eval_data = {
                        "question": [user_query],
                        "answer": [full_answer],
                        "contexts": [[doc.page_content for doc in retrieved_docs]]
                    }
                    eval_dataset = Dataset.from_dict(eval_data)
                    ragas_llm = OllamaLLMWrapper(model_name="llama3")
                    
                    result = evaluate(
                        dataset=eval_dataset,
                        metrics=[faithfulness, answer_relevance],
                        llm=ragas_llm,
                        embeddings=embeddings,
                        raise_exceptions=False
                    )
                    
                    faith_score = result['faithfulness'].iloc[0] if hasattr(result['faithfulness'], 'iloc') else result['faithfulness']
                    rel_score = result['answer_relevance'].iloc[0] if hasattr(result['answer_relevance'], 'iloc') else result['answer_relevance']
                    
                    faith_emoji = "🟢" if faith_score >= 0.8 else "🟡" if faith_score >= 0.6 else "🔴"
                    rel_emoji = "🟢" if rel_score >= 0.8 else "🟡" if rel_score >= 0.6 else "🔴"
                    
                    c1, c2 = st.columns(2)
                    c1.metric(f"{faith_emoji} Faithfulness", f"{faith_score:.1%}", help="100% = no hallucination")
                    c2.metric(f"{rel_emoji} Relevance", f"{rel_score:.1%}", help="100% = answered the question")
                    
                    if faith_score >= 0.8 and rel_score >= 0.8:
                        st.success("✅ **High Confidence**: Well-grounded and relevant.")
                    elif faith_score >= 0.6:
                        st.warning("⚠️ **Moderate Confidence**: Some claims may not be fully supported.")
                    else:
                        st.error("❌ **Low Confidence**: May contain hallucinations or missed the point.")
                        
                except Exception as e:
                    st.warning(f"RAGAS failed: {str(e)[:100]}")
        else:
            st.info("ℹ️ RAGAS disabled. Check terminal for install instructions.")
    
    except ConnectionError:
        st.error("❌ Cannot connect to Ollama. Run: `ollama serve`")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.warning("👈 Ask a question from the sidebar to begin!")

# ==========================================
# 4. CONVERSATION HISTORY
# ==========================================
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Conversation History")
    
    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Q{i}: {item['question'][:70]}..."):
            st.write(item['answer'])
            st.caption(f"📎 {len(item['docs'])} chunks retrieved")