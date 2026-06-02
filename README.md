# HelpmeAtOS - Operating Systems Learning Platform

A comprehensive web-based platform for learning Operating Systems concepts with interactive visualizations, algorithm demonstrations, and an AI-powered chatbot.

🌐 **Live Website**: https://helpmeatos-production.up.railway.app

## Features

### 📚 Learning Modules
- **CPU Scheduling** - Visualize and compare different CPU scheduling algorithms (FCFS, SJF, SRTF, Round Robin, Priority)
- **Memory Management** - Understand memory allocation and management techniques
- **Page Replacement** - Learn about different page replacement strategies
- **Disk Scheduling** - Explore various disk I/O scheduling algorithms

### 💬 AI Chatbot
- Ask questions about Operating Systems concepts
- Get instant, contextual answers powered by Gemini AI
- Works in development mode with sample responses (no API key needed!)

### 📊 Interactive Visualizations
- Gantt charts for CPU scheduling
- Memory allocation diagrams
- Process queue visualizations
- Real-time algorithm simulations

## Tech Stack

**Backend:**
- Flask (Python web framework)
- Gemini AI API (optional, for intelligent responses)
- Chroma DB Cloud (optional, for knowledge base)

**Frontend:**
- HTML5
- CSS3
- Chart visualizations

**Deployment:**
- Railway.app (current live deployment)

## Getting Started

### For Users

Visit the live website: **https://helpmeatos-production.up.railway.app**

All features are fully functional without any setup required!

### For Developers/Contributors

⭐ **Good news**: You can develop and contribute **without API keys**!

#### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HelpmeAtOS.git
   cd HelpmeAtOS
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run locally**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

## Development Mode Features

The application is designed to work **without API keys**, making it perfect for contributors:

✅ **All UI features work**
- Page rendering
- Algorithm visualizations  
- All interactive features

✅ **Chatbot works with sample responses**
- Ask about CPU scheduling, memory management, page replacement, disk scheduling
- Get helpful sample responses

## Adding API Keys (Optional)

To enable AI-powered responses:

1. Copy `.env.example` to `.env`
2. Add your API keys:

```env
# Gemini AI (https://aistudio.google.com)
GEMINI_API_KEY=your_key_here

# Chroma DB Cloud (https://www.trychroma.com)
CHROMA_CLOUD_API_KEY=your_key_here
CHROMA_CLOUD_TENANT=your_tenant_id
```

Restart the app and AI features will be enabled!

## Project Structure

```
HelpmeAtOS/
├── app.py                 # Main Flask application
├── create_knowledge_base.py # Knowledge base setup
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── static/               # Static files (CSS, JS, images)
│   ├── chatbot.css
│   ├── common.css
│   ├── index.css
│   ├── navbar.css
│   └── data/
├── templates/            # HTML templates
│   ├── index.html
│   ├── chatbot.html
│   ├── cpu_scheduling.html
│   ├── memory_management.html
│   ├── page_replacement.html
│   ├── disk_scheduling.html
│   └── navbar.html
├── CONTRIBUTING.md       # Contribution guidelines
├── LICENSE              # Project license
└── README.md            # This file
```

## Contributing

We welcome contributions! Whether you want to:
- Improve the UI/UX
- Add new algorithms
- Enhance visualizations
- Improve documentation
- Fix bugs

**See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.**

### Quick Contribution Tips
1. No API keys needed to get started
2. All scheduling algorithms are in `app.py`
3. Frontend code is in `templates/` and `static/`
4. Test locally before submitting a PR

## Supported Algorithms

### CPU Scheduling
- ✅ First Come First Served (FCFS)
- ✅ Shortest Job First (SJF)
- ✅ Shortest Remaining Time First (SRTF)
- ✅ Round Robin (RR)
- ✅ Priority Scheduling

### Page Replacement
- ✅ FIFO (First In First Out)
- ✅ LRU (Least Recently Used)
- ✅ Optimal Page Replacement

### Disk Scheduling
- ✅ FCFS
- ✅ SCAN
- ✅ C-SCAN
- ✅ LOOK
- ✅ C-LOOK

## FAQ

**Q: Do I need API keys to run the app locally?**
A: No! The app works perfectly without them. You'll get sample responses in the chatbot, and all visualizations work normally.

**Q: How can I contribute?**
A: See [CONTRIBUTING.md](CONTRIBUTING.md) - you can start contributing immediately without any API keys.

**Q: Is the app production-ready?**
A: Yes! It's deployed live on Railway.app and handles production traffic. See https://helpmeatos-production.up.railway.app

**Q: Can I deploy my own instance?**
A: Absolutely! The code is open source. You can deploy to Railway, Heroku, or any Python-hosting platform. Add API keys in `.env` for full features.

**Q: What algorithms can I add?**
A: You can add new scheduling algorithms in `app.py`. See the existing implementations for reference.

## Performance & Rate Limiting

- Default rate limit: 10 requests per minute per IP
- Configurable via `.env` file
- Protects against abuse while allowing active development

## Browser Support

- Chrome/Chromium (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

See [LICENSE](LICENSE) file for details.

## Contact & Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for feature requests
- **Live Site**: https://helpmeatos-production.up.railway.app

## Roadmap

- [ ] More scheduling algorithms
- [ ] Advanced visualizations with animations
- [ ] Practice questions and quizzes
- [ ] User progress tracking
- [ ] Mobile app
- [ ] Multi-language support

## Acknowledgments

- Built with Flask and vanilla JavaScript
- Powered by Gemini AI (when configured)
- Knowledge base via Chroma DB Cloud
- Deployed on Railway.app

---

**Ready to contribute?** Start with [CONTRIBUTING.md](CONTRIBUTING.md) - no API keys required! 🚀

**Want to use the app?** Visit https://helpmeatos-production.up.railway.app 🌐
