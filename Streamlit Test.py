import streamlit as st
#텍스트를 여러 청크로 나눌 때, 문자의 개수를 무엇을 기준으로 산정할 것이냐를 토큰으로 정할 것이므로, 토큰관련 라이브러리 
import tiktoken
#구동한 것이 Streamlit 상 로그로 남도록
from loguru import logger
#메모리를 가지고 있는 Conver~~~Chain 가져오기
from langchain.chains import ConversationalRetrievalChain
#LLM : OpenAI
from langchain.chat_models import ChatOpenAI
#PDF Loader
from langchain.document_loaders import PyPDFLoader
#문서 Loader
from langchain.document_loaders import Docx2txtLoader
#PPT Loader
from langchain.document_loaders import UnstructuredPowerPointLoader
#텍스트 나누는 것
from langchain.text_splitter import RecursiveCharacterTextSplitter
#허깅페이스 임베딩모델 (한국어 특화)
from langchain.embeddings import HuggingFaceEmbeddings
#몇개까지의 대화를 메모리로 넣어줄지 정하는 부분
from langchain.memory import ConversationBufferMemory
#벡터스토어 (임시 저장)
from langchain.vectorstores import FAISS
# from streamlit_chat import message
#메모리 구축을 위한 추가적 라이브러리
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

    st.title("_Kimmin's :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

#채팅창의 메세지들이 계속 유지되기 위한 코드
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

#메세지들마다 Role에 따라 마크다운 할것이다. (한 번 메세지가 입력될 때마다 묶는 것)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


#히스토리를 구현해줘야 우리의 질문을 기억해서 댇답해줌
    history = StreamlitChatMessageHistory(key="chat_messages")

#질문 창 만들기
    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            chain = st.session_state.conversation
#로딩할 때 빙글빙글 돌아가는 영역 구현
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                #답변에 대한 참고문서를 접고 평고 할 수 있음
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


#토큰 개수를 기준으로 Text를 Split 해주기 위한 함수
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

#업로드 된 파일들을 전부 텍스트로 바꾸는 작업
def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

#여러개의 Chunk들로 Split하는 과정
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

#Vector화 해주는 것
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

#위에서 선언한 모든것들을 담아보는 것
def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain

if __name__ == '__main__':
    main()  