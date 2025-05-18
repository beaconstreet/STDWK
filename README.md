# STDWK Content Management Tools

A pair of Python tools built to automate content creation and distribution for **Shit To Do With Kids**, a project that highlights family-friendly events across Los Angeles.

## 📰 Newsletter Writer

Generates region-specific email newsletters using event data from a CSV file.

**Key Features:**

- Groups events by LA neighborhood (e.g., Hollywood, The Valley, South Bay)
- Creates HTML email drafts using a custom template
- Integrates with ConvertKit API to generate and schedule newsletters
- Integrates with OpenAI API to generate appropriate emoji icons for events headlines.

## 📱 Social Media Poster

Produces branded social media content for each event with minimal manual effort.

**Key Features:**

- Generates shareable PNG images with region-specific overlays
- Outputs a scheduler-ready CSV for posting
- Auto-uploads all content to a review site via Netlify

## 🔧 Tech Used

- Python (automation + scripting)
- OpenAI API (text + image generation)
- ConvertKit API (email automation)
- Netlify (static site hosting)
- Pillow, pandas, python-docx, dotenv
