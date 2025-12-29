import streamlit as st

# App title
st.title("Citation Intent Classification (Demo)")
st.write(
    "This is a placeholder Streamlit demo for your Applied ML project. "
    "You can input a citation sentence and get a predicted intent once the model is added."
)

# Text input
citation_text = st.text_area(
    "Enter a citation sentence:",
    height=150,
    placeholder="Example: This method follows the approach proposed by Smith et al. (2020)."
)

# Predict button
if st.button("Predict Intent"):
    if citation_text.strip() == "":
        st.warning("Please enter a citation sentence.")
    else:
        # Placeholder prediction logic
        # You can replace this with your actual model later
        import random
        possible_labels = ["Background", "Method", "Result"]
        prediction = random.choice(possible_labels)
        confidence = round(random.uniform(0.7, 0.99), 2)

        st.success(f"Predicted Citation Intent: **{prediction}**")
        st.write(f"Confidence Score (simulated): {confidence}")