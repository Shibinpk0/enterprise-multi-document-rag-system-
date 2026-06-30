    import os
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    import warnings

    warnings.filterwarnings("ignore")

    print("🚀 Starting Multi-Document Enterprise RAG Ingestion...")

    # 1. Define the folder containing your PDFs
    data_folder = "./data"

    if not os.path.exists(data_folder):
        print(f"❌ Error: '{data_folder}' folder not found! Please create it and add your PDFs.")
        exit()

    # Find all PDF files in the folder
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"❌ Error: No PDF files found in the '{data_folder}' folder!")
        exit()

    print(f"📂 Found {len(pdf_files)} PDF files to process:")
    for f in pdf_files:
        print(f"   - {f}")

    # 2. Load ALL documents from ALL PDFs
    all_documents = []
    for pdf_file in pdf_files:
        file_path = os.path.join(data_folder, pdf_file)
        print(f"\n📖 Reading {pdf_file}...")
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Crucial: Tag every single page with its source filename
            for doc in docs:
                doc.metadata["source_filename"] = pdf_file 
                
            all_documents.extend(docs)
            print(f"   ✅ Extracted {len(docs)} pages.")
        except Exception as e:
            print(f"   ❌ Failed to read {pdf_file}: {e}")

    print(f"\n📚 Total pages loaded across all files: {len(all_documents)}")

    # 3. Chunk the text
    print("✂️ Chunking text into 1000-character pieces...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"✅ Created {len(chunks)} total text chunks.")

    # 4. Create Embeddings & Store in Vector Database
    print("🧠 Loading AI Embedding model...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 Saving all chunks to ChromaDB vector database...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )

    print("\n🎉 SUCCESS! Multi-Document Ingestion complete.")
    print("You can now run: streamlit run app.py")