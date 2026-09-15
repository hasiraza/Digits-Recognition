from time import sleep
from core .logger import logger
import matplotlib.pyplot as plt
import streamlit as st
from core import  model
st.set_page_config(page_title='My Models')
st.set_page_config(layout='wide')
st.title('Digits Recognition')


with st.sidebar:
    st.header("Digits Recognition")
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    **Haseeb Raza**  
    Machine Learning Engineer

    📧 **Email:** [hasiraza511@gmail.com](mailto:hasiraza511@gmail.com)
   # Model Information
    **Algorithm:** SVC  
    **Task:** Image Classification

    **Dataset:** Scikit-learn Digits Dataset
    """)
    st.markdown("---")
    st.subheader("Hyperparameters")
    st.image(
        "public/hyperparameters.png",
        caption="SVC Hyperparameters",
        use_container_width=True
    )
    st.markdown("---")
    st.markdown("---")
    st.subheader("📊 Classification Report")
    st.image(
        "public/classification report.png",
        caption="SVC Classification Report",
        use_container_width=True
    )
    st.markdown("---")

    st.caption("Developed by Haseeb Raza")




uploaded_file = st.file_uploader("Choose a file",   type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    logger.info(f"Uploaded file: {uploaded_file.name}")
    if st.button("predict"):
        with st.spinner("Loading model..."):
            sleep(5)
        col1,col2=st.columns(2)
        with col1:
            st.header("Original Image")
            st.image(uploaded_file, width=250)

        with col2:
            st.header("Model View Image")
            img = model.show_digits(uploaded_file.getvalue())
            fig, ax = plt.subplots(figsize=(2,2))
            ax.imshow(
                img,
                cmap="gray",
                interpolation="nearest"
            )
            ax.axis("off")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
    prediction = model.predict(uploaded_file.getvalue())
    st.success(f"Predicted digit: {prediction}")


