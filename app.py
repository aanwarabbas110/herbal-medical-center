import base64
import io
import json
import os
import requests
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

# --- 1. Page Configuration & Custom CSS Medical Theme ---
st.set_page_config(
    page_title="Herbal Medical Center | AI Diagnostics",
    page_icon="🌿",
    layout="centered",
)

# Custom CSS for Professional Herbal Hospital Theme
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f8f5 0%, #e8f3ec 100%);
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .hospital-header {
        background: linear-gradient(135deg, #0f5132 0%, #198754 100%);
        padding: 28px 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(15, 81, 50, 0.15);
        margin-bottom: 25px;
    }
    .hospital-title {
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .hospital-subtitle {
        font-size: 14px;
        color: #d1e7dd;
        margin-top: 6px;
        font-weight: 400;
    }
    .doctor-badge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.2);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        margin-top: 12px;
        letter-spacing: 0.5px;
    }

    .diagnosis-card {
        background-color: #ffffff;
        border-left: 6px solid #198754;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .diagnosis-title {
        color: #0f5132;
        font-size: 13px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .diagnosis-value {
        color: #198754;
        font-size: 24px;
        font-weight: 800;
    }

    .warning-card {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    .stButton > button {
        background-color: #198754;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0f5132;
        color: white;
        box-shadow: 0 4px 12px rgba(15, 81, 50, 0.3);
    }
    
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar Configuration (Center Information Top Par) ---
st.sidebar.markdown("### 🏥 Center Information")
st.sidebar.markdown("**Lead Developer:** Anwar Ali")
st.sidebar.markdown("**Department:** Botanical Machine Learning & AI Health")
st.sidebar.caption("v3.5 | Streamlit Cloud Ready Edition")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Language & Settings")

language_choice = st.sidebar.selectbox(
    "Select Interface Language / zaban muntakhib karein:",
    ["Roman Urdu", "اردو (Urdu Script)", "English"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚕️ Clinical System Controls")

openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY", "")

if not openrouter_api_key:
  openrouter_api_key = st.sidebar.text_input(
      "Enter OpenRouter API Key:",
      type="password",
      help="Get key from openrouter.ai",
  )

selected_model = "openai/gpt-3.5-turbo"
st.sidebar.info(f"<b>Engine:</b> {selected_model}", icon="🤖")

# Language Map Helper
lang_instructions = {
    "Roman Urdu": (
        "Respond in simple, clear, and professional Roman Urdu. Provide actionable"
        " medical precautions and instructions."
    ),
    "اردو (Urdu Script)": (
        "Respond in proper Urdu script (اردو زبان). Use clean, professional,"
        " and clear phrasing for medical instructions."
    ),
    "English": (
        "Respond in professional, clear, and structured medical English."
    ),
}

current_lang_inst = lang_instructions[language_choice]

# Header Titles
if language_choice == "اردو (Urdu Script)":
  header_title = "🌿 انور علی ہربل میڈیکل سینٹر"
  header_subtitle = "اے آئی سے لیس جڑی بوٹیوں کی تشخیص اور طبی راہنمائی کا نظام"
  doctor_tag = "⚕ چیف ڈائریکٹر: <b>انور علی</b>"
elif language_choice == "English":
  header_title = "🌿 ANWAR ALI HERBAL MEDICAL CENTER"
  header_subtitle = (
      "AI-Powered Botanical Diagnostics & Clinical Guidance System"
  )
  doctor_tag = "⚕️ Chief Director: <b>Anwar Ali</b>"
else:
  header_title = "🌿 ANWAR ALI HERBAL MEDICAL CENTER"
  header_subtitle = (
      "AI-Powered Botanical Diagnostics & Clinical Guidance System"
  )
  doctor_tag = "⚕️ Chief Director: <b>Anwar Ali</b>"

# --- 2. Hospital Header ---
st.markdown(
    f"""
    <div class="hospital-header">
        <div class="hospital-title">{header_title}</div>
        <div class="hospital-subtitle">{header_subtitle}</div>
        <div class="doctor-badge">{doctor_tag}</div>
    </div>
""",
    unsafe_allow_html=True,
)


# Helper function to call OpenRouter API
def call_openrouter(api_key, model_name, prompt):
  headers = {
      "Authorization": f"Bearer {api_key}",
      "Content-Type": "application/json",
      "HTTP-Referer": "http://localhost:8501",
      "X-Title": "Herbal Medical Center Diagnostics",
  }

  messages = [{"role": "user", "content": prompt}]
  payload = {"model": model_name, "messages": messages}

  try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    res_json = response.json()
    if "choices" in res_json and len(res_json["choices"]) > 0:
      return res_json["choices"][0]["message"]["content"]
    elif "error" in res_json:
      return f"API Diagnostic Error: {res_json['error']['message']}"
    else:
      return "Unexpected response structure from Diagnostic Engine."
  except Exception as e:
    return f"Network Diagnostic Failure: {str(e)}"


# --- 3. Data & Model Loading ---
@st.cache_data
def load_data():
  plants_info = {}
  if os.path.exists("plants.json"):
    with open("plants.json", "r", encoding="utf-8") as f:
      plants_info = json.load(f)

  classes = []
  if os.path.exists("classes.json"):
    with open("classes.json", "r", encoding="utf-8") as f:
      classes = json.load(f)

  return plants_info, classes


plant_data, class_names = load_data()


@st.cache_resource
def load_model(num_classes):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = efficientnet_b0(weights=None)
  num_ftrs = model.classifier[1].in_features
  model.classifier[1] = nn.Linear(num_ftrs, num_classes)

  model_path = "herbal_plant_model.pth"
  if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device, True
  return None, device, False


model, device, model_loaded = (
    load_model(len(class_names)) if class_names else (None, "cpu", False)
)

# --- Updated Image Transforms Pipeline for PyTorch 2.2+ Compatibility ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

predicted_plant = None

# --- 4. Main Medical Workspace Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Image Diagnosis",
    "🔍 Disease Search",
    "⚖️ Dosage Calculator",
    "⚠️ Drug Interaction Alert",
    "📲 WhatsApp Order",
])

# ================= TAB 1: IMAGE DIAGNOSTICS =================
with tab1:
  st.markdown("#### 🔬 Patient Sample Submission")
  uploaded_file = st.file_uploader(
      "Upload Leaf Sample Image for Automated Diagnosis (JPG/PNG)...",
      type=["jpg", "jpeg", "png"],
      key="leaf_uploader",
  )

  confidence = 0.0

  if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
      st.image(
          image, caption="Submitted Botanical Specimen", use_container_width=True
      )

    if model_loaded:
      try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
          outputs = model(img_tensor)
          probs = torch.softmax(outputs, dim=1)
          top_prob, top_idx = torch.max(probs, dim=1)

        predicted_plant = class_names[top_idx.item()]
        confidence = top_prob.item() * 100

        with col2:
          st.markdown(
              f"""
                      <div class="diagnosis-card">
                          <div class="diagnosis-title">Diagnostic Identification</div>
                          <div class="diagnosis-value">{predicted_plant.upper()}</div>
                          <p style="margin-top:10px; color:#555;">Classification Confidence: <b>{confidence:.1f}%</b></p>
                      </div>
                  """,
              unsafe_allow_html=True,
          )
      except Exception as e:
        st.error(f"Image Preprocessing Failure: {str(e)}")

  if openrouter_api_key:
    st.markdown("---")
    st.markdown("### 📋 Automated Clinical Botanical Report")

    if predicted_plant:
      with st.spinner("Analyzing pharmacological properties..."):
        prompt = f"""
                  You are acting as an expert Clinical Botanist & Hakim at Anwar Ali Herbal Medical Center.
                  Language Style Instruction: {current_lang_inst}

                  The specimen was identified as '{predicted_plant}' with {confidence:.1f}% diagnostic confidence.
                  Provide a structured report covering:
                  1. Plant Overview & Local Names.
                  2. Medical Uses & Therapeutic Indications (Ilaaj).
                  3. Recommended Preparation Method & Standard Dosage.
                  4. Safety Precautions, Contraindications, & Side Effects.
                  """
        res_text = call_openrouter(openrouter_api_key, selected_model, prompt)
        st.info(res_text)

    # --- Text Chat Area ---
    st.markdown("---")
    st.markdown("### 💬 Medical Consultation Chat")

    user_question = st.text_input(
        "Sawal likhein (e.g., Isay BP ke mariz use kar sakte hain?):",
        key="chat_input",
    )

    if user_question:
      with st.spinner("Consulting Medical Knowledge Base..."):
        chat_prompt = f"""
                  You are an expert Herbal Medical Consultant at Anwar Ali Herbal Medical Center.
                  Language Instruction: {current_lang_inst}
                  Context Plant: '{predicted_plant or 'General Herbal Specimen'}'.
                  Patient Query: {user_question}
                  """
        chat_response = call_openrouter(
            openrouter_api_key, selected_model, chat_prompt
        )
        st.success(f"**Medical Officer:** {chat_response}")
  else:
    st.warning("💡 Configure OpenRouter API Key in sidebar to enable AI features.")

# ================= TAB 2: DISEASE-BASED SEARCH =================
with tab2:
  st.markdown("#### 🔍 Reverse Herbal Lookup (Search by Illness)")
  disease_query = st.text_input(
      "Enter Disease or Symptom (e.g., Sugar, High Blood Pressure, Cough, Joint Pain, Fever):",
      key="disease_search",
  )

  if disease_query:
    if openrouter_api_key:
      with st.spinner(f"Searching herbal remedies for '{disease_query}'..."):
        search_prompt = f"""
                Act as Chief Botanist at Anwar Ali Herbal Medical Center.
                Language Instruction: {current_lang_inst}
                Patient condition/symptom: '{disease_query}'.

                Provide a clean, structured report listing:
                1. Top 3-4 herbal plants/bootiyan useful for '{disease_query}'.
                2. How each plant helps (Mechanisms & Benefits).
                3. Traditional preparation method.
                4. Key safety precautions & contraindications.
                """
        search_res = call_openrouter(
            openrouter_api_key, selected_model, search_prompt
        )
        st.markdown(search_res)
    else:
      st.warning("💡 Enter OpenRouter API Key in sidebar to perform AI Reverse Lookup.")

# ================= TAB 3: DOSAGE & AGE CALCULATOR =================
with tab3:
  st.markdown("#### ⚖️ Interactive Patient Dosage & Safety Calculator")

  c1, c2 = st.columns(2)
  with c1:
    selected_herb = st.selectbox(
        "Select Herbal Plant:",
        class_names if class_names else ["Neem", "Tulsi", "Aloe Vera", "Mint", "Ginger", "Turmeric"],
        key="herb_calc_select",
    )
    patient_age = st.slider("Patient Age (Years):", 1, 90, 25, key="age_slider")

  with c2:
    patient_weight = st.number_input(
        "Patient Weight (kg):", min_value=5, max_value=150, value=65, key="weight_input"
    )
    health_status = st.selectbox(
        "Special Health Condition:",
        [
            "Normal / Healthy",
            "Pregnancy / Lactating",
            "High Blood Pressure",
            "Diabetes",
            "Kidney / Liver Issues",
            "Pediatric (Under 12)",
        ],
        key="condition_select",
    )

  if st.button("Calculate Safe Dosage & Guidance 🧪"):
    if openrouter_api_key:
      with st.spinner("Calculating clinical dosage parameters..."):
        dosage_prompt = f"""
                You are a senior Pharmacologist at Anwar Ali Herbal Medical Center.
                Language Instruction: {current_lang_inst}
                Parameters:
                - Plant: '{selected_herb}'
                - Patient Age: {patient_age} years
                - Patient Weight: {patient_weight} kg
                - Health Status: {health_status}

                Provide precise clinical guidance:
                1. Recommended Safe Dosage.
                2. Administration Timing.
                3. Strict Safety Warnings given age & condition.
                4. Maximum duration of safe use.
                """
        dosage_res = call_openrouter(
            openrouter_api_key, selected_model, dosage_prompt
        )
        st.info(dosage_res)
    else:
      st.warning("💡 Enter OpenRouter API Key in sidebar to calculate dosages.")

# ================= TAB 4: DRUG INTERACTION & ALLERGY WARNING =================
with tab4:
  st.markdown("#### ⚠️ Drug Interaction & Allergy Safety Checker")
  st.write(
      "Check if taking a specific herb alongside your allopathic medicines or medical conditions causes dangerous reactions."
  )

  col_a, col_b = st.columns(2)

  with col_a:
    check_herb = st.selectbox(
        "Select Herbal Plant to Check:",
        class_names if class_names else ["Neem", "Tulsi", "Aloe Vera", "Mint", "Ginger", "Turmeric"],
        key="check_herb_select",
    )

    existing_conditions = st.multiselect(
        "Existing Conditions / Allergies:",
        [
            "Pregnancy / Nursing",
            "Hypertension (High BP)",
            "Hypotension (Low BP)",
            "Diabetes",
            "Kidney Disease",
            "Liver Disease",
            "Bleeding / Clotting Disorder",
            "Pollen Allergy",
        ],
        key="conditions_check",
    )

  with col_b:
    current_meds = st.multiselect(
        "Current Allopathic Medicines:",
        [
            "Blood Thinners (Aspirin, Warfarin)",
            "Diabetes Meds / Insulin",
            "BP Blood Pressure Meds",
            "Sedatives / Anti-Anxiety",
            "Painkillers / NSAIDs (Panadol, Ibuprofen)",
            "Antacids / Proton Pump Inhibitors",
            "Immunosuppressants",
        ],
        key="meds_check",
    )

  if st.button("Analyze Drug-Herb Interactions 🛡"):
    if openrouter_api_key:
      with st.spinner("Checking pharmacological interaction matrix..."):
        meds_str = ", ".join(current_meds) if current_meds else "None reported"
        cond_str = (
            ", ".join(existing_conditions)
            if existing_conditions
            else "None reported"
        )

        interaction_prompt = f"""
                You are Chief Clinical Toxicologist & Hakim at Anwar Ali Herbal Medical Center.
                Language Instruction: {current_lang_inst}
                Evaluate interaction safety:
                - Herb: '{check_herb}'
                - Current Medicines: {meds_str}
                - Existing Health Conditions/Allergies: {cond_str}

                Provide a safety assessment:
                1. Safety Verdict: (SAFE / USE WITH CAUTION / DANGEROUS CONTRAINDICATION).
                2. Potential Interactions or Adverse Effects between '{check_herb}' and listed medicines/conditions.
                3. Toxicological risks or symptoms to watch out for.
                4. Final Recommendation by Anwar Ali Medical Board.
                """
        interaction_res = call_openrouter(
            openrouter_api_key, selected_model, interaction_prompt
        )

        st.markdown(
            f"""
            <div class="warning-card">
                <h4 style="color:#856404; margin-top:0;">🛡 Pharmacological Safety Report ({check_herb})</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(interaction_res)
    else:
      st.warning("💡 Enter OpenRouter API Key in sidebar to perform interaction analysis.")

# ================= TAB 5: DIRECT WHATSAPP CONSULTATION & ORDER =================
with tab5:
  st.markdown("#### 📲 Direct WhatsApp Order & Medical Consultation")
  st.write(
      "Apni zaroorat ya jari booti (herb) ki tafseelat neeche darj karein aur"
      " direct WhatsApp par Anwar Ali Herbal Medical Center se rabta karein."
  )

  with st.form("whatsapp_order_form"):
    p_name = st.text_input("Patient Name / Naam:", key="wa_name")
    p_phone = st.text_input("Phone Number / WhatsApp No:", key="wa_phone")
    herb_needed = st.text_input(
        "Required Herb / Disease Detail (Jari Booti ya Bimari):",
        value=predicted_plant if predicted_plant else "",
        key="wa_herb",
    )
    user_msg = st.text_area(
        "Message / Masla (Optional):",
        placeholder="Apna masla ya order ki miqdar likhein...",
        key="wa_msg",
    )

    submit_btn = st.form_submit_button("Start WhatsApp Chat 💬")

    if submit_btn:
      if p_name and p_phone:
        full_message = (
            f"🌿 *ANWAR ALI HERBAL MEDICAL CENTER*\n"
            f"-------------------------------------\n"
            f"👤 *Patient:* {p_name}\n"
            f"📞 *Contact:* {p_phone}\n"
            f"🌱 *Item/Condition:* {herb_needed if herb_needed else 'General Consultation'}\n"
            f"💬 *Detail:* {user_msg if user_msg else 'Direct consultation required'}\n"
            f"-------------------------------------\n"
            f"Sent from AI Diagnostics Portal"
        )

        whatsapp_number = "923065440740"  # Pre-configured contact number
        encoded_msg = requests.utils.quote(full_message)
        whatsapp_url = (
            f"https://wa.me/{whatsapp_number}?text={encoded_msg}"
        )

        st.success("Tafseelat tayar hain! Neeche diye gaye button par click karke WhatsApp open karein.")
        st.markdown(
            f"""
            <a href="{whatsapp_url}" target="_blank" style="text-decoration:none;">
                <button style="
                    background-color: #25D366;
                    color: white;
                    padding: 12px 24px;
                    border: none;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 16px;
                    cursor: pointer;
                    width: 100%;
                    margin-top: 10px;">
                    📲 Open WhatsApp Chat Now
                </button>
            </a>
            """,
            unsafe_allow_html=True,
        )
      else:
        st.error("Lazmi fields (Name aur Phone Number) fill karein.")