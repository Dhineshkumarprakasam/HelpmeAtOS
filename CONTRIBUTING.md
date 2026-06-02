# Contributing to HelpmeAtOS

Thank you for your interest in contributing to HelpmeAtOS! This guide will help you set up the development environment and get started.

## Development Setup (Without API Keys)

The app is now designed to work **without API keys**, making it perfect for UI and feature development. Here's how to set it up:

### Prerequisites
- Python 3.8+
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HelpmeAtOS.git
   cd HelpmeAtOS
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app without API keys**
   ```bash
   python app.py
   ```
   
   The app will start in **development mode** and display messages like:
   ```
   Info: GEMINI_API_KEY not set in .env file
   Info: App running in development mode without Gemini API - chatbot will use sample responses
   Info: Chroma DB Cloud not available
   Info: App running in development mode - knowledge base features disabled
   ```

5. **Open in your browser**
   ```
   http://localhost:5000
   ```

## What Works Without API Keys

✅ **All UI/Frontend Features:**
- All pages render correctly
- CPU Scheduling visualizations
- Memory Management features
- Page Replacement algorithms
- Disk Scheduling algorithms

✅ **Chatbot UI:**
- Chatbot interface loads and is fully functional
- Sample responses are provided for OS-related questions

## What Requires API Keys (Optional)

If you want to enable AI-powered responses:

1. **Get a Gemini API Key:**
   - Go to [Google AI Studio](https://aistudio.google.com)
   - Create an API key
   - Create a `.env` file in the project root:
     ```env
     GEMINI_API_KEY=your_api_key_here
     ```

2. **Get Chroma DB Cloud Access (Optional):**
   - Go to [Chroma](https://www.trychroma.com)
   - Set up your cloud instance
   - Add to `.env`:
     ```env
     CHROMA_CLOUD_API_KEY=your_api_key
     CHROMA_CLOUD_TENANT=your_tenant
     CHROMA_CLOUD_DATABASE=operating_system
     CHROMA_COLLECTION_NAME=os-knowledge-base
     ```

## Development Tips

### Running in Debug Mode
```bash
export FLASK_ENV=development  # On Windows: set FLASK_ENV=development
export FLASK_DEBUG=1          # On Windows: set FLASK_DEBUG=1
python app.py
```

### Making Changes
- **Frontend changes**: Edit files in `templates/` and `static/`
- **Backend changes**: Edit `app.py` or create new routes
- **Algorithms**: Add new scheduling algorithms in `app.py`

### Testing the Chatbot (Dev Mode)
The chatbot works with sample responses. Try asking about:
- "What is CPU scheduling?"
- "Explain memory management"
- "How does page replacement work?"
- "Tell me about disk scheduling"

## Common Areas for Contribution

### UI/Frontend
- Improve styling in `static/` (CSS files)
- Enhance visualization of algorithms
- Improve responsive design for mobile

### Features
- Add more scheduling algorithms
- Create interactive visualizations
- Add practice questions

### Backend
- Optimize algorithm implementations
- Add caching mechanisms
- Improve error handling

### Documentation
- Write better algorithm explanations
- Add code comments
- Create tutorial pages

## Getting Help

- Check existing issues on GitHub
- Read the README.md for project overview
- Run the app and explore the UI

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test thoroughly (no API keys required!)
5. Commit with clear messages (`git commit -m 'Add feature: ...'`)
6. Push to your fork
7. Open a Pull Request

## Code Style

- Follow PEP 8 for Python code
- Use clear variable and function names
- Add comments for complex logic
- Test your changes before submitting

## Questions?

Feel free to open an issue or start a discussion in the repository!

Happy contributing! 🚀
