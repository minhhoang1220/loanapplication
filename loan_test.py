import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
from streamlit_chat import message
from datetime import datetime 

# Upload model
model = joblib.load('random_forest_pipeline.pkl')

def predict_loan_status(model, user_data):
    predicted_result = model.predict(user_data)
    probability = model.predict_proba(user_data)[0][1]
    return predicted_result[0], probability

# Load CSS from style.css
css_path = r'style.css'
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_header():
    st.markdown('<div class="header"><h1>Loan Application Service</h1></div>', unsafe_allow_html=True)

def show_products():
    st.markdown('<div class="centered-title"><h2>Recommended Products</h2></div>', unsafe_allow_html=True)
    st.write("Here are some products that might interest you based on your profile and preferences:")
    products = [
        {"name": "Product 1", "description": "This is a great product for those who need A."},
        {"name": "Product 2", "description": "This product is ideal if you are looking for B."},
        {"name": "Product 3", "description": "Consider this product if you are interested in C."}
    ]
    for product in products:
        st.markdown(f"### {product['name']}")
        st.write(product['description'])

@st.experimental_dialog(title="Customer Information Form")
def customer_info_dialog():
    with st.form("personal_info_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents (0-No / 1-Yes)", ["0", "1"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
        credit_history = st.selectbox("Credit History", [1.0, 0.0])
        loan_amount_term = st.number_input("Loan Amount Term", min_value=12, max_value=360, step=12)
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        submit_personal_info = st.form_submit_button("Submit Personal Info")
    
    if submit_personal_info:
        total_income = applicant_income + coapplicant_income
        st.session_state.personal_info = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
            "Credit_History": credit_history,
            "Loan_Amount_Term": loan_amount_term,
            "Total_Income": total_income,
            "Loan_Amount": loan_amount
        }
        st.write("Personal information submitted successfully!")
        save_to_csv(st.session_state.personal_info)
        st.rerun()

def save_to_csv(personal_info):
    df = pd.DataFrame([personal_info])
    file_path = "personal_info.csv"
    try:
        existing_data = pd.read_csv(file_path)
        df = pd.concat([existing_data, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(file_path, index=False)

def loan_calculation_dialog():
    st.markdown('<div class="centered-title"><h2>Loan Calculation</h2></div>', unsafe_allow_html=True)
    st.write("Adjust the loan amount and term to see different repayment options:")
    loan_amount = st.slider("Loan Amount", 1000, 50000, 20000)
    loan_term = st.slider("Loan Term (years)", 1, 30, 5)
    interest_rate = 5  # Placeholder value
    monthly_payment = (loan_amount * (1 + (interest_rate / 100) * loan_term)) / (loan_term * 12)
    st.write(f"Monthly Payment: ${monthly_payment:.2f}")

# First login
def login():
    st.title("User Login")

    st.write("Please enter your basic information:")

    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")

    if st.button("Submit"):
        if name and email and phone:
            st.success(f"Welcome, {name}!")
            st.write(f"Email: {email}")
            st.write(f"Phone Number: {phone}")
            st.session_state.logged_in = True
            st.session_state.user_name = name
            st.session_state.user_email = email
            st.session_state.user_phone = phone
            st.rerun()
        else:
            st.error("Please fill in all the fields")

def profile():
    st.markdown('<div class="centered-title"><h2>Personal Information</h2></div>', unsafe_allow_html=True)
    personal_info_df = pd.DataFrame([st.session_state.personal_info])
    st.table(personal_info_df)

    user_data = pd.DataFrame([st.session_state.personal_info])
    predicted_result, probability = predict_loan_status(model, user_data)

    st.markdown('<div class="centered-title"><h2>Prediction Results</h2></div>', unsafe_allow_html=True)
    if predicted_result == 1:
        st.success(f"**Predicted Loan Status:** Approved", icon=":material/thumb_up:")
    else:
        st.warning(f"**Predicted Loan Status:** Rejected", icon=":material/thumb_down:")
    st.write(f"**Approval Probability:** {probability:.2%}")

    loan_amount = st.session_state.personal_info['Loan_Amount']
    interest_rate = 5  # Placeholder value
    loan_term = st.session_state.personal_info['Loan_Amount_Term']

    st.write(f"**Loan Amount:** ${loan_amount}")
    st.write(f"**Interest Rate:** {interest_rate}%")
    st.write(f"**Loan Term:** {loan_term} months")

    with st.sidebar:
        st.markdown("---")  # Horizontal line
        st.header("Chat")
        chat()

def chat():
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    def generate_response(user_input):
        if "interest rate" in user_input.lower():
            return "The current interest rate is 5%."
        elif "approved" in user_input.lower():
            return "I can help you check the approval status."
        elif "calculate" in user_input.lower():
            return "Your loan is approved based on a machine model learned from user history and that data has been thoroughly filtered and processed."
        else:
            return "I'm here to assist you with your loan queries."

    def on_input_change():
        user_input = st.session_state.user_input
        st.session_state.past.append(user_input)
        response = generate_response(user_input)
        st.session_state.generated.append(response)
        st.session_state.user_input = ""  # Reset user input

    def on_btn_click():
        del st.session_state.past[:]
        del st.session_state.generated[:]

    chat_placeholder = st.empty()

    with chat_placeholder.container():    
        for i in range(len(st.session_state['generated'])):                
            message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
            message(st.session_state['generated'][i], key=f"{i}", allow_html=True)
        
        st.button("Clear message", on_click=on_btn_click)

    st.text_input("User Input:", on_change=on_input_change, key="user_input")

# Main function
def main():
    if 'logged_in' not in st.session_state:
        login()
    else:
        with st.sidebar:
            st.markdown(f'<p class="sidebar-name">{st.session_state.user_name}</p>', unsafe_allow_html=True)  # Display name in bold and larger font size
            current_time = datetime.now().strftime("Upd. %b %d, %Y %I:%M %p")
            st.markdown(f'<p class="sidebar-updated">{current_time}</p>', unsafe_allow_html=True)  # Display date and time in smaller font size and black color
            st.markdown("---")  # Horizontal line
            st.markdown(
                f"""
                <div class="sidebar-box">
                    <p><strong>Email:</strong> {st.session_state.user_email}</p>
                    <p><strong>Phone:</strong> {st.session_state.user_phone}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        selected = option_menu(
            None, ["Profile", "Loan Calculation", "Recommended Products"], 
            icons=["person", "calculator", "gift"], 
            menu_icon="cast", default_index=0, orientation="horizontal"
        )
        
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        show_header()
        
        if 'personal_info' not in st.session_state:
            customer_info_dialog()
        else:
            if selected == "Profile":
                profile()
            elif selected == "Loan Calculation":
                loan_calculation_dialog()
            elif selected == "Recommended Products":
                show_products()
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
