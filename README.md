# X-pressionist Telegram Bot

A Telegram bot that analyzes X (Twitter) accounts and generates satirical AI-powered content including artwork, videos, roasts, and profiles.

## Features

- **/pic @username** - Generate satirical cartoon/artwork with style selection
- **/video @username** - Generate animated satirical video
- **/roast @username** - Generate comedy roast letter
- **/fbi @username** - Generate satirical FBI behavioral profile
- **/osint @username** - Generate OSINT-style dossier
- **/joint @user1 @user2** - Generate image featuring two accounts together
- **/caricature** - Reply to a photo to generate a caricature

## Image Generation

Two AI models available for image generation:

- **Nano Banana Pro** - OpenRouter + Google Gemini (default)
- **Grok Imagine** - xAI's native image generation

12 art styles to choose from:
Cartoon, Realistic, Anime, Cyberpunk, Fantasy, Cinematic, Watercolor, Oil Painting, Minimalist, Surreal, Gothic, Retro

## Setup

### 1. Install dependencies

```bash
pip install python-telegram-bot httpx python-dotenv
```

### 2. Create `.env` file

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
XAI_API_KEY=your_xai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 3. Get API keys

- **Telegram Bot Token**: Create a bot via [@BotFather](https://t.me/BotFather)
- **xAI API Key**: Get from [x.ai](https://x.ai)
- **OpenRouter API Key**: Get from [openrouter.ai](https://openrouter.ai)

### 4. Run the bot

```bash
python bot.py
```

## Tech Stack

- **python-telegram-bot** - Telegram Bot API wrapper
- **httpx** - Async HTTP client
- **xAI Grok** - Account analysis with X search capability
- **xAI Grok Imagine** - Native image/video generation
- **OpenRouter + Gemini** - Alternative image generation

## How It Works

1. User sends a command with an X username
2. Bot uses xAI's Grok model with X search to analyze the account
3. Grok generates a creative prompt based on the account's posts and personality
4. The prompt is sent to an image generation model
5. The resulting artwork is sent back to the user

## License

MIT
