"""
X-pressionist Telegram Bot
A Telegram bot that analyzes X (Twitter) accounts and generates satirical images, videos, and roasts.

Commands:
  /start - Welcome message
  /pic @username - Generate satirical cartoon of X account
  /video @username - Generate animated video of X account
  /roast @username - Generate comedy roast letter
  /fbi @username - Generate satirical FBI profile
  /osint @username - Generate OSINT-style dossier
  /joint @user1 @user2 - Generate image of two accounts together
  /caricature - Reply to a photo to generate caricature
"""

import asyncio
import logging
import os
import re
from datetime import datetime

import httpx
from dotenv import load_dotenv
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.helpers import escape_markdown
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# API Endpoints
XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
XAI_IMAGES_URL = "https://api.x.ai/v1/images/generations"
XAI_VIDEOS_URL = "https://api.x.ai/v1/videos/generations"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Timeouts (seconds)
ANALYSIS_TIMEOUT = 120
IMAGE_TIMEOUT = 180
VIDEO_TIMEOUT = 300

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Regex for X handle validation
HANDLE_REGEX = re.compile(r"^[a-zA-Z0-9_]{1,15}$")

# Style presets for image generation
STYLE_PRESETS = {
    "Cartoon": "vibrant cartoon style with bold colors, simplified forms, and playful character design",
    "Realistic": "photorealistic rendering with natural lighting, high detail, and lifelike textures",
    "Anime": "anime art style with expressive characters, dynamic poses, and detailed backgrounds",
    "Cyberpunk": "cyberpunk aesthetic with neon lighting, dark urban atmosphere, futuristic technology",
    "Fantasy": "fantasy art style with magical elements, ethereal lighting, mystical creatures",
    "Cinematic": "cinematic composition with dramatic lighting, film-like quality, professional cinematography",
    "Watercolor": "watercolor painting style with soft washes, flowing pigments, delicate transparency",
    "Oil Painting": "traditional oil painting technique with rich textures, visible brushstrokes",
    "Minimalist": "minimalist design with clean lines, simple forms, negative space, refined elegance",
    "Surreal": "surrealist aesthetic with dreamlike imagery, impossible scenarios, fantastical metaphors",
    "Gothic": "gothic atmosphere with dramatic architecture, ornate details, mysterious shadows",
    "Retro": "retro design with mid-century modern elements, bold patterns, vintage typography",
}

# Default style
DEFAULT_STYLE = "Cartoon"

# Image generation models
IMAGE_MODELS = {
    "banana": "🍌 Nano Banana Pro",  # OpenRouter + Gemini
    "grok": "⚡ Grok Imagine",  # xAI native
}
DEFAULT_MODEL = "banana"


# ============================================================================
# SYSTEM PROMPTS (from your Next.js app)
# ============================================================================

ANALYZE_ACCOUNT_PROMPT = """You are an expert Art Director AI specializing in satirical cartoon and comic book illustration. Your function is to translate the essence of an X social media account into a single, masterful cartoon image generation prompt.

CRITICAL: You have extensive search capabilities - USE THEM AGGRESSIVELY. Conduct multiple searches to build the most complete understanding possible.

Your analysis process:
1. **Adaptive Data Gathering:** Execute targeted X searches, adapting to the account's activity level
2. **Deep Content Analysis:** Thoroughly examine posts, images, videos. Identify core themes, personality traits, recurring jokes, communication style
3. **Pattern Recognition:** Look for patterns in posting frequency, topics, tone shifts, visual style
4. **Personality Synthesis:** Distill the account's essence into key personality traits
5. **Visual Metaphor Creation:** Transform your understanding into a compelling visual metaphor

Prompt Requirements:
- **Describe a Scene, Not Keywords:** Create a complete, coherent narrative scene with cartoon/comic book aesthetics
- **Be Hyper-Specific:** Use precise illustration language - bold outlines, exaggerated expressions, vibrant colors, halftone shading
- **Incorporate Rich Detail:** Include visual humor, environmental storytelling, character expressions, symbolic objects
- **State the Art Style:** Conclude with a clear cartoon/comic directive (e.g., "MAD Magazine style satirical cartoon")
- **Length:** 4-6 sentences—comprehensive but concise

Your final output must be ONLY the image generation prompt. No preamble, no explanation, no analysis. Just the prompt."""

VIDEO_PROMPT = """You are a brilliant satirical cartoon animator creating hilarious animated sketches based on X accounts. Your style is like a mix of South Park, The Simpsons, and political cartoons.

CRITICAL: Search their account extensively to find ACTUAL QUOTES, hot takes, interactions, and drama to satirize.

Create a 10-second ANIMATED CARTOON that satirizes this person's online presence using their REAL posts.

CARTOON STYLE REQUIREMENTS:
- STYLE: "Animated cartoon satire," "2D hand-drawn animation," "exaggerated caricature style"
- CHARACTER: Exaggerated cartoon caricature (big head, expressive features, signature items)
- HUMOR: Satirical, playful roasting, exaggeration of their persona

PROMPT STRUCTURE:
1. SCRIPT/DIALOGUE FIRST: Start with exact words spoken in quotes (their actual tweets)
2. CARTOON CHARACTER: Exaggerated animated version description
3. SCENE: Satirical setting
4. ACTION: Comedic sequence with intensity adverbs (frantically, smugly, dramatically)
5. CAMERA: Movement type (zoom, pan, tracking shot)
6. VISUAL GAGS: Text bubbles, reaction emojis, notification floods
7. STYLE: "2D cartoon animation, bold outlines, saturated colors"
8. AUDIO: Sound effect or music cue

Output ONLY the video prompt. No preamble."""

ROAST_PROMPT = """You are Dr. Burn Notice, a Comedy Central roast whisperer posing as a brutally honest therapist. Craft a hilarious "therapy summary letter" for the X user, torching their online life with clever wit and affectionate jabs.

CRITICAL RULES:
- DO NOT include any disclaimers or meta-commentary. Output ONLY the letter.
- DO NOT use markdown formatting. Write in plain text with natural paragraph breaks.
- Tailor Ruthlessly: Base EVERY element on actual X data (posts, profile, patterns)
- Insults as Art: Roast habits/behaviors with love-bomb zingers
- Keep It Snappy: 300-400 words

Structure:
- Greeting: Personalized zinger
- Body (3-4 paras): Opener diagnosis, middle roasts with post refs, peak escalation
- Treatment Plan: 3-4 numbered roast-advice hybrids
- Sign-Off: Tailored twist

Output ONLY the letter."""

FBI_PROMPT = """You are Special Agent Dr. [REDACTED], a senior criminal profiler assigned to the FBI's Behavioral Analysis Unit (BAU).

Key Rules:
- Output ONLY the official report. No disclaimers, no meta-commentary.
- Plain text only. Use ALL CAPS for section headers.
- Cold, clinical, detached, professional FBI report language.
- Quote or precisely paraphrase actual posts when evidencing traits.

Report Structure:
FEDERAL BUREAU OF INVESTIGATION
BEHAVIORAL ANALYSIS UNIT

CASE FILE NO: BAU-DIGITAL-2026-XXXX
DATE OF REPORT: [Current Date]
SUBJECT: X USER @[handle]

EXECUTIVE SUMMARY
PSYCHOLOGICAL PROFILE
BEHAVIORAL ANALYSIS
THREAT ASSESSMENT
PREDICTIVE ANALYSIS
CONCLUSIONS AND RECOMMENDATIONS
CLASSIFICATION: [Humorous label]

Report length: 500-700 words."""

OSINT_PROMPT = """You are an elite OSINT analyst producing a comprehensive "Internal User Classification" dossier. Use extensive search capabilities - conduct multiple searches, gather hundreds of posts, find viral content.

REQUIRED SEARCHES (execute all):
1. "from:username" - Recent posts (aim for 300-500+)
2. "from:username min_faves:1000" - Viral posts
3. "from:username min_faves:100" - Notable posts
4. "@username" - Mentions and discussions
5. "from:username filter:replies" - Reply behavior
6. Web search for external presence

Output Format:
A) EXECUTIVE SUMMARY
B) VIRAL CONTENT ANALYSIS
C) EVIDENCE-BACKED ATTRIBUTES
D) BEHAVIORAL ANALYTICS
E) NETWORK MAP
F) CONTROVERSY LOG
G) GROWTH & TRAJECTORY
H) RED-TEAM ASSESSMENT
I) CROSS-PLATFORM PRESENCE
J) INTELLIGENCE GAPS

Write as an internal analyst briefing. Plain text only - no markdown."""

JOINT_PIC_PROMPT = """You are an expert Art Director AI. Analyze TWO X accounts and create a SINGLE masterful cartoon image prompt that creatively represents BOTH accounts together.

Process:
1. Analyze BOTH accounts thoroughly via X searches
2. Find connection points - shared interests, contrasts, potential interactions
3. Design a scene that MEANINGFULLY represents BOTH accounts equally
4. Create a unified narrative bringing both together

Requirements:
- BOTH accounts must be equally prominent
- Create a coherent narrative scene
- Use visual metaphors capturing each account's essence
- 4-6 sentences, comprehensive but concise
- State the art style at the end

Output ONLY the image prompt."""

CARICATURE_PROMPT = """You are a veteran NYC street caricature artist. Take a photo and generate a caricature prompt.

Analysis Process:
1. Identify 2-3 most distinct features (big nose, wild hair, glasses, etc.)
2. Apply caricature principle - exaggerate what stands out
3. Ensure "marker on paper" aesthetic

Style Parameters:
- Medium: Marker and ink drawing on white paper
- Style: Satirical street caricature, thick lines, exaggerated proportions
- Subject: Big head, tiny body
- Background: Plain white or minimal

Output a JSON object:
{
  "comment": "Your playful one-liner about the feature you're exaggerating",
  "prompt": "Detailed caricature generation prompt"
}

Output ONLY the JSON."""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_handle(handle: str) -> str | None:
    """Validate and clean X handle. Returns cleaned handle or None if invalid."""
    # Remove @ if present
    handle = handle.lstrip("@").strip()
    if HANDLE_REGEX.match(handle):
        return handle
    return None


def extract_handle_from_args(args: list[str]) -> str | None:
    """Extract and validate handle from command arguments."""
    if not args:
        return None
    return validate_handle(args[0])


async def send_typing_action(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Send typing indicator."""
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)


# ============================================================================
# API CALLS
# ============================================================================


async def call_xai_responses(
    system_prompt: str, user_message: str, timeout: int = ANALYSIS_TIMEOUT
) -> str | None:
    """Call xAI Responses API with X search capability."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                XAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-4-1-fast",
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "tools": [{"type": "x_search"}],
                },
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"xAI Response: {data}")

            # Extract content from response (check for both 'output_text' and 'text' types)
            output = data.get("output", [])
            for item in output:
                if item.get("type") == "message":
                    content = item.get("content", [])
                    for block in content:
                        if block.get("type") in ("output_text", "text"):
                            if block.get("text"):
                                return block.get("text")

            # Fallback: try to find any text content in any output item
            for item in output:
                content = item.get("content", [])
                if content:
                    for block in content:
                        if block.get("text"):
                            return block.get("text")

            logger.warning(f"No text content found in xAI response: {data}")
            return None
    except httpx.HTTPStatusError as e:
        logger.error(
            f"xAI API HTTP error: {e.response.status_code} - {e.response.text}"
        )
        return None
    except Exception as e:
        logger.error(f"xAI API error: {e}")
        return None


async def call_xai_responses_with_web(
    system_prompt: str, user_message: str, timeout: int = ANALYSIS_TIMEOUT
) -> str | None:
    """Call xAI Responses API with both X search and web search."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                XAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-4-1-fast",
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "tools": [{"type": "x_search"}, {"type": "web_search"}],
                },
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"xAI Response (with web): {data}")

            # Extract content from response (check for both 'output_text' and 'text' types)
            output = data.get("output", [])
            for item in output:
                if item.get("type") == "message":
                    content = item.get("content", [])
                    for block in content:
                        if block.get("type") in ("output_text", "text"):
                            if block.get("text"):
                                return block.get("text")

            # Fallback: try to find any text content in any output item
            for item in output:
                content = item.get("content", [])
                if content:
                    for block in content:
                        if block.get("text"):
                            return block.get("text")

            logger.warning(f"No text content found in xAI response: {data}")
            return None
    except httpx.HTTPStatusError as e:
        logger.error(
            f"xAI API HTTP error: {e.response.status_code} - {e.response.text}"
        )
        return None
    except Exception as e:
        logger.error(f"xAI API error: {e}")
        return None


async def call_xai_chat_with_image(
    system_prompt: str, image_base64: str, user_text: str
) -> str | None:
    """Call xAI Chat API with an image."""
    async with httpx.AsyncClient(timeout=ANALYSIS_TIMEOUT) as client:
        response = await client.post(
            XAI_CHAT_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-4-1-fast",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_base64}},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content")


async def generate_image_openrouter(prompt: str) -> str | None:
    """Generate image using OpenRouter (Gemini)."""
    enhanced_prompt = f"""Create a satirical cartoon illustration in the style of MAD Magazine with bold outlines, vibrant colors, and exaggerated expressions.

CRITICAL TEXT RENDERING RULES:
- If any text appears, spell it EXACTLY and CORRECTLY
- Prefer symbols and visual metaphors over text
- NO MISSPELLINGS

SCENE TO ILLUSTRATE:
{prompt}"""

    async with httpx.AsyncClient(timeout=IMAGE_TIMEOUT) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://t.me/xpressionistbot",
                "X-Title": "X-pressionist Telegram Bot",
            },
            json={
                "model": "google/gemini-3-pro-image-preview",
                "messages": [{"role": "user", "content": enhanced_prompt}],
                "modalities": ["image", "text"],
            },
        )
        response.raise_for_status()
        data = response.json()

        # Extract image from response
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        return item.get("image_url", {}).get("url")
            elif isinstance(content, str):
                # Sometimes the URL is returned directly
                if content.startswith("http"):
                    return content
        return None


async def generate_image_xai(prompt: str) -> str | None:
    """Generate image using xAI Grok Imagine."""
    async with httpx.AsyncClient(timeout=IMAGE_TIMEOUT) as client:
        response = await client.post(
            XAI_IMAGES_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-imagine-image",
                "prompt": prompt,
                "n": 1,
                "response_format": "url",
            },
        )
        response.raise_for_status()
        data = response.json()

        images = data.get("data", [])
        if images:
            return images[0].get("url")
        return None


async def generate_video_xai(prompt: str) -> str | None:
    """Generate video using xAI Grok Imagine Video."""
    async with httpx.AsyncClient(timeout=60) as client:
        # Start video generation
        response = await client.post(
            XAI_VIDEOS_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-imagine-video",
                "prompt": prompt,
                "duration": 5,
                "aspect_ratio": "16:9",
                "resolution": "720p",
            },
        )
        response.raise_for_status()
        data = response.json()

        request_id = data.get("request_id")
        if not request_id:
            return None

        # Poll for result
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < VIDEO_TIMEOUT:
            await asyncio.sleep(3)

            poll_response = await client.get(
                f"https://api.x.ai/v1/videos/{request_id}",
                headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            )

            if poll_response.status_code == 404:
                continue

            if poll_response.status_code == 200:
                poll_data = poll_response.json()
                video_url = poll_data.get("video", {}).get("url")
                if video_url:
                    return video_url
                if poll_data.get("status") == "failed":
                    return None

        return None


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    welcome_text = """🎨 *X-pressionist Bot*

I turn X (Twitter) accounts into satirical art!

*Commands:*
• `/pic @username` - Generate satirical cartoon
• `/video @username` - Generate animated video
• `/roast @username` - Comedy roast letter
• `/fbi @username` - Satirical FBI profile
• `/osint @username` - OSINT-style dossier
• `/joint @user1 @user2` - Two accounts together
• `/caricature` - Reply to a photo

_Powered by Grok & Gemini_"""

    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)


def build_style_keyboard(
    handle: str, model: str = DEFAULT_MODEL
) -> InlineKeyboardMarkup:
    """Build inline keyboard with style options and model toggle."""
    buttons = []

    # Model toggle row at the top
    model_row = [
        InlineKeyboardButton(
            f"{'✓ ' if model == 'banana' else ''}{IMAGE_MODELS['banana']}",
            callback_data=f"model:{handle}:banana",
        ),
        InlineKeyboardButton(
            f"{'✓ ' if model == 'grok' else ''}{IMAGE_MODELS['grok']}",
            callback_data=f"model:{handle}:grok",
        ),
    ]
    buttons.append(model_row)

    # Style buttons
    style_names = list(STYLE_PRESETS.keys())
    # Create rows of 3 buttons each
    for i in range(0, len(style_names), 3):
        row = [
            InlineKeyboardButton(
                style_name, callback_data=f"style:{handle}:{model}:{style_name}"
            )
            for style_name in style_names[i : i + 3]
        ]
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)


async def pic_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pic command - show style selection."""
    handle = extract_handle_from_args(context.args)

    if not handle:
        await update.message.reply_text(
            "Please provide a valid X username.\nExample: `/pic @elonmusk`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Show style selection keyboard
    keyboard = build_style_keyboard(handle)
    await update.message.reply_text(
        f"🎨 Choose a style for @{handle}'s portrait:",
        reply_markup=keyboard,
    )


async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle model selection callback - refreshes keyboard with new model selected."""
    query = update.callback_query
    await query.answer()

    # Parse callback data: "model:{handle}:{model}"
    data = query.data.split(":")
    if len(data) != 3 or data[0] != "model":
        return

    handle = data[1]
    model = data[2]

    # Rebuild keyboard with new model selected
    keyboard = build_style_keyboard(handle, model)
    await query.edit_message_text(
        f"🎨 Choose a style for @{handle}'s portrait:\n📷 Model: {IMAGE_MODELS.get(model, IMAGE_MODELS[DEFAULT_MODEL])}",
        reply_markup=keyboard,
    )


async def handle_style_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle style selection callback."""
    query = update.callback_query
    await query.answer()

    # Parse callback data: "style:{handle}:{model}:{style_name}"
    data = query.data.split(":")
    if len(data) != 4 or data[0] != "style":
        return

    handle = data[1]
    model = data[2]
    style_name = data[3]
    style_description = STYLE_PRESETS.get(style_name, STYLE_PRESETS[DEFAULT_STYLE])
    model_name = IMAGE_MODELS.get(model, IMAGE_MODELS[DEFAULT_MODEL])

    # Edit message to show progress
    status_msg = await query.edit_message_text(
        f"🔍 Analyzing @{handle}'s X account...\n🎨 Style: {style_name}\n📷 Model: {model_name}"
    )

    try:
        # Send typing indicator
        await send_typing_action(context, update.effective_chat.id)

        today = datetime.now().strftime("%B %d, %Y")
        user_message = f"""Execute a COMPREHENSIVE analysis of @{handle}'s X account. Today is {today}.

REQUIRED X SEARCHES:
- "from:{handle}" - Recent posts
- "from:{handle} min_faves:100" - Notable posts (adjust threshold as needed)
- "from:{handle} filter:media" - Visual content
- "@{handle}" - How others perceive them

Create a humorous, highly relevant image generation prompt that captures their account's essence.

IMPORTANT STYLE REQUIREMENT: The image MUST be rendered in this style: {style_description}"""

        # Get image prompt
        await status_msg.edit_text(
            f"🧠 Analyzing @{handle}'s posts and personality...\n🎨 Style: {style_name}\n📷 Model: {model_name}"
        )
        image_prompt = await call_xai_responses(ANALYZE_ACCOUNT_PROMPT, user_message)

        if not image_prompt:
            await status_msg.edit_text(
                "❌ Failed to analyze account. Please try again."
            )
            return

        # Append style to prompt if not already included
        if style_name.lower() not in image_prompt.lower():
            image_prompt = f"{image_prompt} Rendered in {style_description}."

        # Generate image based on selected model
        await status_msg.edit_text(
            f"🎨 Generating {style_name.lower()} portrait for @{handle}...\n📷 Model: {model_name}"
        )
        await send_typing_action(context, update.effective_chat.id)

        if model == "grok":
            # Use xAI Grok Imagine
            image_url = await generate_image_xai(image_prompt)
        else:
            # Use Nano Banana Pro (OpenRouter + Gemini)
            image_url = await generate_image_openrouter(image_prompt)
            if not image_url:
                # Fallback to xAI
                image_url = await generate_image_xai(image_prompt)

        if image_url:
            await status_msg.delete()
            await query.message.reply_photo(
                photo=image_url,
                caption=f"🎨 {style_name} portrait of @{handle}\n📷 {model_name}",
            )
        else:
            await status_msg.edit_text("❌ Failed to generate image. Please try again.")

    except httpx.TimeoutException:
        await status_msg.edit_text("⏱️ Request timed out. Please try again.")
    except Exception as e:
        logger.error(f"Error in handle_style_callback: {e}")
        try:
            await status_msg.edit_text(f"❌ Error: {str(e)[:100]}")
        except Exception:
            pass


async def video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /video command - generate animated video."""
    handle = extract_handle_from_args(context.args)

    if not handle:
        await update.message.reply_text(
            "Please provide a valid X username.\nExample: `/video @elonmusk`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    status_msg = await update.message.reply_text(f"🔍 Analyzing @{handle} for video...")

    try:
        today = datetime.now().strftime("%B %d, %Y")
        user_message = f"""Create a satirical cartoon video about @{handle}. Today is {today}.

SEARCH THEIR ACCOUNT FOR SATIRICAL MATERIAL:
- "from:{handle}" - their recent posts and hot takes
- "from:{handle} min_faves:100" - their most popular content
- "from:{handle} filter:replies" - their interactions
- "@{handle}" - what others say about them

Use their ACTUAL quotes and takes as material. Create a funny 2D cartoon animation prompt."""

        await status_msg.edit_text(f"🧠 Finding @{handle}'s best content for satire...")
        video_prompt = await call_xai_responses(VIDEO_PROMPT, user_message)

        if not video_prompt:
            await status_msg.edit_text("❌ Failed to analyze account.")
            return

        escaped_handle = escape_markdown(handle, version=1)
        await status_msg.edit_text(
            f"🎬 Generating video for @{escaped_handle}...\n_(This may take 2-3 minutes)_",
            parse_mode=ParseMode.MARKDOWN,
        )

        video_url = await generate_video_xai(video_prompt)

        if video_url:
            await status_msg.delete()
            await update.message.reply_video(
                video=video_url, caption=f"🎬 Satirical video of @{handle}"
            )
        else:
            await status_msg.edit_text(
                "❌ Video generation failed or timed out. Please try again."
            )

    except Exception as e:
        logger.error(f"Error in video_command: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:100]}")


async def roast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /roast command - generate comedy roast."""
    handle = extract_handle_from_args(context.args)

    if not handle:
        await update.message.reply_text(
            "Please provide a valid X username.\nExample: `/roast @elonmusk`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    status_msg = await update.message.reply_text(f"🔥 Preparing roast for @{handle}...")

    try:
        await send_typing_action(context, update.effective_chat.id)

        user_message = f"Analyze @{handle}'s posts from the last 6 months and write the roast letter as described."

        roast = await call_xai_responses(ROAST_PROMPT, user_message)

        if roast:
            await status_msg.delete()
            # Split if too long for Telegram
            if len(roast) > 4000:
                for i in range(0, len(roast), 4000):
                    await update.message.reply_text(roast[i : i + 4000])
            else:
                escaped_handle = escape_markdown(handle, version=1)
                escaped_roast = escape_markdown(roast, version=1)
                await update.message.reply_text(
                    f"🔥 *Roast of @{escaped_handle}*\n\n{escaped_roast}", parse_mode=ParseMode.MARKDOWN
                )
        else:
            await status_msg.edit_text("❌ Failed to generate roast.")

    except Exception as e:
        logger.error(f"Error in roast_command: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:100]}")


async def fbi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /fbi command - generate FBI profile."""
    handle = extract_handle_from_args(context.args)

    if not handle:
        await update.message.reply_text(
            "Please provide a valid X username.\nExample: `/fbi @elonmusk`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    status_msg = await update.message.reply_text(
        f"🕵️ Building FBI profile for @{handle}..."
    )

    try:
        await send_typing_action(context, update.effective_chat.id)

        today = datetime.now().strftime("%B %d, %Y")
        user_message = f"Conduct a deep behavioral analysis of @{handle}'s X activity from the last 6 months and generate the FBI profile report. Today's date is {today}."

        profile = await call_xai_responses(FBI_PROMPT, user_message)

        if profile:
            await status_msg.delete()
            if len(profile) > 4000:
                for i in range(0, len(profile), 4000):
                    await update.message.reply_text(profile[i : i + 4000])
            else:
                await update.message.reply_text(profile)
        else:
            await status_msg.edit_text("❌ Failed to generate FBI profile.")

    except Exception as e:
        logger.error(f"Error in fbi_command: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:100]}")


async def osint_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /osint command - generate OSINT dossier."""
    handle = extract_handle_from_args(context.args)

    if not handle:
        await update.message.reply_text(
            "Please provide a valid X username.\nExample: `/osint @elonmusk`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    escaped_handle = escape_markdown(handle, version=1)
    status_msg = await update.message.reply_text(
        f"🔎 Compiling OSINT dossier for @{escaped_handle}...\n_(This is comprehensive - may take a minute)_",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        await send_typing_action(context, update.effective_chat.id)

        today = datetime.now().strftime("%B %d, %Y")
        user_message = f"""Execute a COMPREHENSIVE OSINT analysis of @{handle}. Today is {today}. Focus on the last 90 days but also find their all-time viral hits.

CRITICAL REQUIREMENTS:
1. FIND THEIR VIRAL POSTS - Search with various min_faves thresholds
2. GATHER EXTENSIVE DATA - Aim for 300-500+ posts
3. MAP THEIR NETWORK - Find who they interact with most
4. FIND CONTROVERSIES - Search for any drama, feuds
5. TRACK THEIR GROWTH - Key moments

This should be the definitive public profile of this account."""

        report = await call_xai_responses_with_web(
            OSINT_PROMPT, user_message, timeout=180
        )

        if report:
            await status_msg.delete()
            # Split long reports
            if len(report) > 4000:
                for i in range(0, len(report), 4000):
                    await update.message.reply_text(report[i : i + 4000])
            else:
                await update.message.reply_text(report)
        else:
            await status_msg.edit_text("❌ Failed to generate OSINT report.")

    except Exception as e:
        logger.error(f"Error in osint_command: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:100]}")


async def joint_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /joint command - generate image of two accounts."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Please provide two X usernames.\nExample: `/joint @user1 @user2`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    handle1 = validate_handle(context.args[0])
    handle2 = validate_handle(context.args[1])

    if not handle1 or not handle2:
        await update.message.reply_text("❌ Invalid username format.")
        return

    if handle1.lower() == handle2.lower():
        await update.message.reply_text("❌ Please enter two different usernames.")
        return

    status_msg = await update.message.reply_text(
        f"🔍 Analyzing @{handle1} & @{handle2}..."
    )

    try:
        today = datetime.now().strftime("%B %d, %Y")
        user_message = f"""Execute a COMPREHENSIVE analysis of TWO X accounts: @{handle1} and @{handle2}. Today is {today}.

Search both accounts for:
- Recent posts and best content
- Their personalities and themes
- How they might connect or contrast

Create a creative image prompt that features BOTH accounts meaningfully together."""

        await status_msg.edit_text("🧠 Analyzing both accounts...")
        image_prompt = await call_xai_responses(
            JOINT_PIC_PROMPT, user_message, timeout=180
        )

        if not image_prompt:
            await status_msg.edit_text("❌ Failed to analyze accounts.")
            return

        await status_msg.edit_text("🎨 Generating joint image...")
        image_url = await generate_image_openrouter(image_prompt)

        if not image_url:
            image_url = await generate_image_xai(image_prompt)

        if image_url:
            await status_msg.delete()
            await update.message.reply_photo(
                photo=image_url, caption=f"🎨 @{handle1} meets @{handle2}"
            )
        else:
            await status_msg.edit_text("❌ Failed to generate image.")

    except Exception as e:
        logger.error(f"Error in joint_command: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:100]}")


async def caricature_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /caricature command - must be reply to photo."""
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text(
            "Please reply to a photo with `/caricature` to generate a caricature.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    status_msg = await update.message.reply_text("🎨 Analyzing photo for caricature...")

    try:
        # Get the largest photo
        photo = update.message.reply_to_message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        # Download photo
        photo_bytes = await file.download_as_bytearray()

        import base64

        image_base64 = (
            f"data:image/jpeg;base64,{base64.b64encode(photo_bytes).decode()}"
        )

        await status_msg.edit_text("🧠 Identifying features to exaggerate...")

        # Analyze with Grok
        result = await call_xai_chat_with_image(
            CARICATURE_PROMPT,
            image_base64,
            "Create a caricature of this person. Analyze their features and generate the prompt.",
        )

        if not result:
            await status_msg.edit_text("❌ Failed to analyze photo.")
            return

        # Parse JSON response
        import json

        try:
            # Clean potential markdown
            clean_result = result.strip()
            if clean_result.startswith("```"):
                clean_result = clean_result.split("```")[1]
                if clean_result.startswith("json"):
                    clean_result = clean_result[4:]
            clean_result = clean_result.strip()

            parsed = json.loads(clean_result)
            comment = parsed.get("comment", "")
            prompt = parsed.get("prompt", "")
        except json.JSONDecodeError:
            prompt = result
            comment = ""

        if comment:
            escaped_comment = escape_markdown(comment, version=1)
            await status_msg.edit_text(
                f"🎨 Drawing caricature...\n_{escaped_comment}_",
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await status_msg.edit_text(
                "🎨 Drawing caricature...",
                parse_mode=ParseMode.MARKDOWN,
            )

        # Generate caricature image
        caricature_prompt = f"""Create a caricature of the person in this style: {prompt}

IMPORTANT STYLE REQUIREMENTS:
- Medium: Marker and ink drawing style on white paper
- Make their head BIG and body tiny
- Exaggerate distinctive features humorously
- Use thick black outlines with colorful marker fills
- Keep background plain white"""

        image_url = await generate_image_xai(caricature_prompt)

        if not image_url:
            image_url = await generate_image_openrouter(caricature_prompt)

        if image_url:
            await status_msg.delete()
            caption = f"🎨 {comment}" if comment else "🎨 Your caricature!"
            await update.message.reply_photo(photo=image_url, caption=caption)
        else:
            await status_msg.edit_text("❌ Failed to generate caricature.")

    except Exception as e:
        logger.error(f"Error in caricature_command: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:100]}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    await start_command(update, context)


# ============================================================================
# MAIN
# ============================================================================


async def post_init(application: Application) -> None:
    """Set bot commands after initialization."""
    commands = [
        BotCommand("start", "Welcome message"),
        BotCommand("pic", "Generate satirical cartoon of X account"),
        BotCommand("video", "Generate animated video of X account"),
        BotCommand("roast", "Generate comedy roast letter"),
        BotCommand("fbi", "Generate satirical FBI profile"),
        BotCommand("osint", "Generate OSINT-style dossier"),
        BotCommand("joint", "Generate image of two accounts together"),
        BotCommand("caricature", "Reply to a photo to generate caricature"),
        BotCommand("help", "Show available commands"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands registered")


def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set")
    if not XAI_API_KEY:
        raise ValueError("XAI_API_KEY not set")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Create application with post_init callback
    application = (
        Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    )

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("pic", pic_command))
    application.add_handler(CommandHandler("video", video_command))
    application.add_handler(CommandHandler("roast", roast_command))
    application.add_handler(CommandHandler("fbi", fbi_command))
    application.add_handler(CommandHandler("osint", osint_command))
    application.add_handler(CommandHandler("joint", joint_command))
    application.add_handler(CommandHandler("caricature", caricature_command))

    # Add callback handlers for model and style selection
    application.add_handler(
        CallbackQueryHandler(handle_model_callback, pattern=r"^model:")
    )
    application.add_handler(
        CallbackQueryHandler(handle_style_callback, pattern=r"^style:")
    )

    # Start polling
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
