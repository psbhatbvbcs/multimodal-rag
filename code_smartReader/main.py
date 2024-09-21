import streamlit as st
from styles.cssTemplate import css
from controllers import processPdf
from controllers import ragApproach1, ragApproach2
from dotenv import load_dotenv
import multiprocessing
import atexit

# Load environment variables from a .env file
load_dotenv()

# Function to handle the termination of the process


def terminate_process():
    if "process" in st.session_state and st.session_state.process is not None:
        if st.session_state.process.is_alive():
            st.session_state.process.kill()
            st.session_state.process.join()


# Register the terminate_process function to be called on exit
atexit.register(terminate_process)


def main():
    st.header("SmartReader, Team Pheonix")
    st.title("Multimodal Chat! :books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize conversation history in session state if not already done
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "image_history" not in st.session_state:
        st.session_state.image_history = []

    # Initialize the vector database in session state if not already done
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    # Initialize the vector store status in session state if not already done
    if "vector_initialized" not in st.session_state:
        st.session_state.vector_initialized = False

    # Input for user to type their question
    question = st.chat_input("You: ")
    answer = ""
    image_data = {}

    # Sidebar for uploading and processing documents
    with st.sidebar:

        # Add radio button for choosing the approach
        st.markdown("## Choose an approach")
        st.markdown(
            "### NOTE! \nPlease stop the code and rerun in terminal to try out approach 2. Won't work otherwise")
        approach = st.radio("Choice", ("Approach 1", "Approach 2"))

        if approach == "Approach 1":
            with st.popover("Learn more"):
                st.markdown("## Approach 1")
                st.write("""<ul><li><b>Our own quicker method</b></ul>
                            <ul><li>Does not calculate image embeddings, hence faster PDF Processing</ul>
                            <ul><li>Finds most relevant images for the given answer based on similarity search from the page where answer is taken from</ul>
                        """,
                         unsafe_allow_html=True)

        if approach == "Approach 2":
            with st.popover("Learn more"):
                st.markdown("## Approach 2")
                st.write("""<ul><li><b>Conventional but Slower method</b></ul>
                            <ul><li>Calculates image descriptions and embeddings for all images when processing PDF</ul>
                            <ul><li>Output based on description of images, not according to page numbers</ul>
                        """,
                         unsafe_allow_html=True)

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your documents and click 'Process'", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                PDFprocessor = processPdf.PDFprocessor(
                    pdf_docs=pdf_docs, pdf_names=[pdf_doc.name for pdf_doc in pdf_docs])
                chunks = PDFprocessor.parse_pdf()
                csv_chunks = PDFprocessor.get_tables()
                PDFprocessor.extract_images()

                if approach == "Approach 1":
                    # Terminate any existing process
                    st.session_state.process = multiprocessing.Process(
                        target=PDFprocessor.background_upload)
                    st.session_state.process.start()

                print("Embedding started")
                PDFprocessor.embed_images()
                print("Embedding done")

                image_chunks = []

                if approach == "Approach 2":
                    image_chunks = PDFprocessor.get_image_chunks()

                # Display the name of the uploaded PDF
                for pdf in pdf_docs:
                    st.write(f"Uploaded PDF: {pdf.name}")

                if approach == "Approach 1":
                    # Initialize the vector database with the processed PDF and CSV chunks
                    st.session_state.vector_db = ragApproach1.ragApproach1(
                        pdf_docs=pdf_docs, chunks=chunks, pdf_names=[
                            pdf_doc.name for pdf_doc in pdf_docs],
                        csv_chunks=csv_chunks
                    )
                    # Create the vector store
                    st.session_state.vector_initialized = st.session_state.vector_db.create_vector_store()

                elif approach == "Approach 2":
                    # Initialize the vector database with the processed PDF and CSV chunks
                    st.session_state.vector_db = ragApproach2.ragApproach2(
                        pdf_docs=pdf_docs, chunks=chunks, pdf_names=[
                            pdf_doc.name for pdf_doc in pdf_docs],
                        csv_chunks=csv_chunks, image_chunks=image_chunks
                    )
                    # Create the vector store
                    st.session_state.vector_initialized = st.session_state.vector_db.create_vector_store()

    if question:
        if st.session_state.vector_initialized:
            # Update the conversation with the user's question
            st.session_state.conversation.append({
                "role": "user",
                "parts": [question]
            })
            # st.session_state.image_history.append({
            #     "image_data": None
            # })
            # Get the bot's response
            answer, image_data = st.session_state.vector_db.get_answer_for_text_query(
                question=question, chat_history=st.session_state.conversation
            )

            # Update the conversation with the bot's response
            st.session_state.conversation.append({
                "role": "model",
                "parts": [answer]
            })
            st.session_state.image_history.append({
                "image_data": image_data
            })
            question = ""
        else:
            st.error("Please upload a PDF document and enter a question.", icon="🚨")
            st.stop()

    with st.sidebar:
        if st.session_state.vector_initialized:
            # Input for user to upload an image
            st.subheader("Search with image!")
            uploaded_image = st.file_uploader(
                "Upload an image for similarity search", type=["png", "jpg", "jpeg"])

            # Handle image upload for similarity search
            if st.button("Process Image"):
                if uploaded_image:
                    if st.session_state.vector_initialized:
                        PDFprocessor = processPdf.PDFprocessor(
                            pdf_docs=[], pdf_names=[])
                        with st.spinner("Calculating similarity..."):
                            similar_image_path = PDFprocessor.calculate_similarity(
                                uploaded_image)
                            if similar_image_path:
                                if st.session_state.vector_initialized:
                                    # Update the conversation with the user's question
                                    st.session_state.conversation.append({
                                        "role": "user",
                                        "parts": ["Processing Image..."]
                                    })

                                    answer, image_data = st.session_state.vector_db.get_image_context(
                                        question="Processing Image...", similar_image_path=similar_image_path, chat_history=st.session_state.conversation
                                    )
                                    # Update the conversation with the bot's response
                                    st.session_state.conversation.append({
                                        "role": "model",
                                        "parts": [answer]
                                    })
                                    st.session_state.image_history.append({
                                        "image_data": image_data
                                    })
                                    question = ""

                            else:
                                st.write(
                                    "No similar image found with similarity > 0.95")
                    else:
                        st.error(
                            "Please process a PDF document first.", icon="🚨")
                        st.stop()

    # Create a two-column layout
    # col1, col2 = st.columns([2, 1])

    # with col1:
    # Display the conversation in the left column
    # st.markdown('<div class="conversation-container">',
    #             unsafe_allow_html=True)  # Start conversation container
    count = 0
    for message in st.session_state.conversation:
        # Ignore messages with user_context
        if message["parts"][0].startswith("TEXTUAL CONTEXT: "):
            continue

        if message["role"] == "user":
            role = "User"
        elif message["role"] == "model":
            role = "Bot"

        # Choose the template based on the role (user or bot)
        srole = "human" if role == "User" else "ai"
        with st.chat_message(srole):
            st.markdown(message["parts"][0])
            if srole == "ai":
                try:
                    if st.session_state.image_history[count]:
                        for image_path, caption in st.session_state.image_history[count]["image_data"].items():
                            st.image(image_path, caption=caption)
                    else:
                        print("\033[91mNone image data\033[0m")
                    count += 1
                except Exception as e:
                    print(f"\033[91m{e}\033[0m")

    # End conversation container
    st.markdown('</div>', unsafe_allow_html=True)


# with col2:
# Display images and captions in the right column
# if image_data:
#     for image_path, caption in image_data.items():
#         st.image(image_path, caption=caption)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        st.stop()
