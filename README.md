 
****🔐 PhoeniksRedTeamers: Ethical LLM Jailbreaking & Red Teaming App 🚀****




**PhoeniksRedTeamers** is an app designed for ethical red teaming, jailbreaking, and educational purposes, providing a controlled environment to explore and understand the behavior of large language models (LLMs).
some of work is inspired by pliny 
https://github.com/elder-plinius
https://x.com/elder_plinius/status/1814023961535295918
---

## 🔧 Features

- **Jailbreak and Test LLMs**: Explore various models for educational purposes.
- **Multi-Model Support**: Interact with models from services like Together AI, OpenAI-like, Groq, and Openrouter.
- **Customizable Experience**: Select different models and services to fit your needs.
- **Intuitive Chat Interface**: Communicate with AI models seamlessly.
- **Text Conversion Tools**: Convert text to formats like leetspeak, base64, binary, and emoji.
- **Prompt Library**: Access a collection of prompts for testing via Prompts.csv.

---

## 🚀 Getting Started

### 1. Clone the Repository:
```bash
git clone https://github.com/sallumandya/PhoeniksRedTeamers.git
```

### 2. Set Up Virtual Environment (Optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
```

### 3. Install Required Packages:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables:

- **Together AI**:  
  `.env` file in the root directory:
  ```
  TOGETHER_API_KEY=<your_together_api_key>
  ```

- **OpenAI-like**:  
  `.env` file:
  ```
  API_KEY=<your_openai_api_key>
  API_BASE=<your_openai_base_url>
  ```

- **Groq**:  
  `.env` file:
  ```
  GROQ_API_KEY=<your_groq_api_key>
  ```

- **Openrouter**:  
  `.env` file:
  ```
  API_KEY_OPENROUTER=<your_openrouter_api_key>
  API_BASE_OPENROUTER=<your_openrouter_base_url>
  ```

### 5. Run the App:
```bash
streamlit run app.py
```

---

## 📝 Usage

1. **Choose a Service**: Select from the sidebar.
2. **Select a Model**: Pick from available options.
3. **Enter API Details**: Input API key and base URL if needed.
4. **Interact**: Type your prompt and send it to see the AI's response.
5. **Text Conversion Tools**: Transform text using sidebar options.
6. **Testing Prompts**: Explore various prompts from Prompts.csv.

---

## 🔒 Security and Responsibility

- **Intended Use**: For educational and safety purposes only.
- **Compliance**: Ensure activities align with the terms of service of the model providers.
- **Legal Notice**: Misuse may result in account suspension or legal action.

---

## 🤝 Contributing

- **Issues and PRs**: Open issues or submit pull requests on GitHub for improvements.

---

## 🌟 Support

- **GitHub**: Star the project if you find it helpful.

---

## 📜 License

- **MIT License**: This project is licensed under the MIT License.

---

## 💻 Developed by
- ** Xhaheen <3 **

Happy jailbreaking and ethical red teaming! 🔐
