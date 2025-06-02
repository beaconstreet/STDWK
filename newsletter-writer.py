from openai import OpenAI
import csv
import shutil
import sys
import requests
from datetime import date, datetime, timedelta
import os
from urllib.parse import urlparse
import mimetypes
import base64
from PIL import Image
import io
from typing import Dict, List, Tuple, Optional
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class ConvertKitIntegration:
    """
    A class to handle ConvertKit integration for the newsletter generator.
    """
    def __init__(self, api_key: str, api_secret: str, template_id: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.template_id = template_id
        self.base_url = "https://api.convertkit.com/v3"

    def send_newsletter(self, content, location, preview=False):
        """
        Sends a newsletter using ConvertKit API v3
        """
        url = f"{self.base_url}/broadcasts"
        
        # Create a proper subject line based on location
        # Special case for westside - use "on" instead of "in"
        if location == "westside":
            subject = f"Things To Do With Kids on {get_location_title(location)} This Week"
        else:
            subject = f"Things To Do With Kids in {get_location_title(location)} This Week"
        
        # Format the data according to ConvertKit's API documentation
        data = {
            "api_secret": self.api_secret,
            "content": content,
            "subject": subject,
            "description": f"What are we doing this weekend?",
            "email_layout_template": self.template_id,
            "public": not preview,
            "published_at": None if preview else datetime.utcnow().isoformat()
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            broadcast_data = response.json()
            
            if not preview:
                broadcast_id = broadcast_data["broadcast"]["id"]
                publish_url = f"{self.base_url}/broadcasts/{broadcast_id}/publish"
                publish_data = {"api_secret": self.api_secret}
                publish_response = requests.post(publish_url, json=publish_data)
                publish_response.raise_for_status()
                print(f"Newsletter sent successfully!")
            else:
                print(f"Newsletter draft created successfully!")
                
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating broadcast: {str(e)}")
            return False

    def get_subscribers_count(self) -> Optional[int]:
        """Gets the total number of active subscribers."""
        try:
            response = requests.get(
                f"{self.base_url}/subscribers",
                params={"api_secret": self.api_secret}
            )
            if response.status_code == 200:
                return response.json()['total_subscribers']
            return None
        except Exception as e:
            print(f"Error getting subscriber count: {str(e)}")
            return None

    def get_templates(self) -> Optional[List[Dict]]:
        """Gets all available email templates."""
        try:
            response = requests.get(
                f"{self.base_url}/templates",
                params={"api_secret": self.api_secret}
            )
            if response.status_code == 200:
                return response.json()['templates']
            return None
        except Exception as e:
            print(f"Error getting templates: {str(e)}")
            return None

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ConvertKit integration (comment out to disable)
convertkit = ConvertKitIntegration(
    api_key=os.getenv("CONVERTKIT_API_KEY"),
    api_secret=os.getenv("CONVERTKIT_API_SECRET"),
    template_id=os.getenv("CONVERTKIT_TEMPLATE_ID")
)

# Settings and constants
MAX_TOKENS = 1_000_000
TOTAL_TOKENS_USED = 0
INPUT_CSV = "events_update.csv"
OUTPUT_CSV = "events_update_newsletter.csv"
ASSETS_FOLDER = "assets/"

# Updated location groups
LOCATION_GROUPS = {
    "downtown": "downtown_eastla",
    "eastla": "downtown_eastla",
    "hollywood": "hollywood_midcity",
    "midcity": "hollywood_midcity",
    "westvalley": "valley",
    "thevalley": "valley",
    "southbay": "southbay_longbeach",
    "longbeach": "southbay_longbeach",
    "westside": "westside"  # Stays separate
}

# Updated locations list (now using group identifiers)
LOCATIONS = ["downtown_eastla", "hollywood_midcity", "valley", "southbay_longbeach", "westside"]

# Group titles for newsletters
GROUP_TITLES = {
    "downtown_eastla": "Downtown + East LA",
    "hollywood_midcity": "Hollywood + Midcity",
    "valley": "The Valley",
    "southbay_longbeach": "The South Bay + Long Beach",
    "westside": "The Westside"
}

# Original location titles (keep for backward compatibility)
LOCATION_TITLES = {
    "downtown": "Downtown",
    "eastla": "East LA",
    "hollywood": "Hollywood",
    "longbeach": "Long Beach",
    "midcity": "Mid City",
    "southbay": "The South Bay",
    "thevalley": "The Valley",
    "westside": "The Westside",
    "westvalley": "West Valley"
}

def get_location_title(location: str) -> str:
    """Convert location slug to proper title format"""
    # If it's a group identifier, use the group title
    if location in GROUP_TITLES:
        return GROUP_TITLES[location]
    # Otherwise use the original location title or capitalize
    return LOCATION_TITLES.get(location.lower(), location.title())

def get_location_group(location: str) -> str:
    """Get the group identifier for a location"""
    return LOCATION_GROUPS.get(location.lower(), location.lower())

# Create assets folder if it doesn't exist
os.makedirs(ASSETS_FOLDER, exist_ok=True)

def get_user_approval(content, content_type):
    """
    Gets user approval for generated content.
    Returns the approved content or None if user wants to end the script.
    """
    while True:
        print(f"\nGenerated {content_type}:")
        print("-" * 50)
        print(content)
        print("-" * 50)
        response = input("Approve this content? (y/n/end): ").lower().strip()
        
        if response == 'y':
            return content
        elif response == 'n':
            return None
        elif response == 'end':
            print("Script terminated by user.")
            sys.exit(0)
        else:
            print("Please enter 'y', 'n', or 'end'")

def generate_event_writeup(event):
    """
    Generates a write-up for a single event using ChatGPT.
    """
    prompt = (
        f"Write a short, 1-2 sentence fun description for parents about the event '{event['activity']}'. "
        f"Include reasons why parents should go. The event is on {event['daysOpen']} and the category is {event['category']}."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write like you're a knowledgeable friend, you can include occasional light humor but prioritize clarity. Include information that might entice them to go to this event."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.4
    )
    global TOTAL_TOKENS_USED
    TOTAL_TOKENS_USED += response.usage.total_tokens
    return response.choices[0].message.content

def download_image(url, activity):
    """
    Downloads an image from a URL and saves it to the assets folder.
    Returns the local path to the saved image or None if failed.
    """
    if not url:  # Add check for empty URL
        print(f"No URL provided for {activity}")
        return None
        
    try:
        # Print URL for debugging
        print(f"Attempting to download image from URL: {url}")
        
        # Use a session with proper headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # First try a HEAD request to check content type
        head_response = session.head(url, allow_redirects=True)
        print(f"HEAD response status code: {head_response.status_code}")
        print(f"HEAD response headers: {dict(head_response.headers)}")
        
        # Then get the actual image
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Get content type and validate it's an image
        content_type = response.headers.get('content-type', '').lower()
        print(f"Content-Type: {content_type}")
        
        if not content_type.startswith('image/'):
            print(f"Warning: Content-Type {content_type} may not be an image")
            
        # Try to get extension from content-type first
        ext = mimetypes.guess_extension(content_type)
        
        # If that fails, try to get it from the URL
        if not ext:
            parsed_url = urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1]
            
        # If all else fails, default to .jpg
        if not ext:
            ext = '.jpg'
            
        print(f"Using file extension: {ext}")
        
        # Create filename from activity (sanitized)
        safe_activity = ''.join(c for c in activity if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        # If the image is webp, we'll convert it to jpg
        if ext.lower() == '.webp':
            filename = f"downloaded_{safe_activity.lower().replace(' ', '_')}.jpg"
        else:
            filename = f"downloaded_{safe_activity.lower().replace(' ', '_')}{ext}"
            
        local_path = os.path.join(ASSETS_FOLDER, filename)
        print(f"Saving to: {local_path}")
        
        # For webp images, convert to jpg
        if ext.lower() == '.webp':
            # Read the webp image into memory
            image_content = response.content
            image = Image.open(io.BytesIO(image_content))
            
            # Convert to RGB if necessary (in case of RGBA webp)
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as JPEG
            image.save(local_path, 'JPEG', quality=95)
        else:
            # Save other formats directly
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
        # Verify file was created and has size
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"Successfully saved image to {local_path}")
            return os.path.join("assets", filename)
        else:
            print(f"Failed to save image or file is empty: {local_path}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request error downloading image for {activity}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error downloading image for {activity}: {str(e)}")
        return None

def generate_dalle_image(description, activity_name):
    """
    Generates an image using DALL-E 3 and returns the local path where it's saved.
    Returns None if generation fails.
    """
    try:
        prompt = f"minimalist, vector illustration of kids enjoying {description}. Do not include any text, letters, or numbers in the artwork. Generate only one image."
        print(f"\nGenerating DALL-E image with prompt: {prompt}")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            style="vivid"
        )
        
        # Get the image URL
        image_url = response.data[0].url
        
        print("\nImage generated! Opening in web browser for review...")
        import webbrowser
        webbrowser.open(image_url)
        
        while True:
            approval = input(f"\nApprove this image for '{activity_name}'? (y/n/end): ").lower().strip()
            if approval == 'y':
                # Download and save the approved image
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                
                today = date.today().strftime("%Y%m%d")
                filename = f"dalle_{description.lower().replace(' ', '_')}_{today}.png"
                local_path = os.path.join(ASSETS_FOLDER, filename)
                
                with open(local_path, 'wb') as f:
                    f.write(image_response.content)
                
                return os.path.join("assets", filename)
            elif approval == 'n':
                print("Generating new image...")
                return generate_dalle_image(description, activity_name)
            elif approval == 'end':
                print("Script terminated by user.")
                sys.exit(0)
            else:
                print("Please enter 'y', 'n', or 'end'")
                
    except Exception as e:
        print(f"Error generating DALL-E image for {description}: {str(e)}")
        return None

def handle_image(event):
    """
    Handles image processing based on imageUrl value.
    Returns the path or URL to use in the HTML.
    """
    image_url = event.get('imageUrl', '')
    
    if not image_url:
        # Generate DALL-E image if no image URL provided
        print(f"Generating DALL-E image for: {event['activity']}")
        image_path = generate_dalle_image(event.get('description', event['activity']), event['activity'])
    elif image_url.lower().startswith('http'):
        # Return the remote URL as-is
        return image_url
    else:
        # Handle local image path
        if image_url.startswith('assets/'):
            # Use the path as-is if it already includes 'assets/'
            local_path = image_url
        else:
            # Otherwise, prepend the assets folder
            local_path = os.path.join(ASSETS_FOLDER, image_url)
            
        if os.path.exists(local_path):
            return local_path if local_path.startswith('assets/') else os.path.join("assets", image_url)
        else:
            print(f"Local image not found for {event['activity']}: {local_path}")
            return None

def create_updated_csv(events, input_file, output_file):
    """
    Creates a new CSV file with updated image URLs.
    """
    if not events:  # Add check for empty events list
        print("No events to update in CSV")
        return
        
    # Read all rows from original CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # Update rows with new image paths
    for event in events:
        for row in rows:
            if (row['activity'] == event['activity'] and 
                row['location'] == event['location']):
                if 'new_image_path' in event:
                    row['imageUrl'] = event['new_image_path']
    
    # Write updated rows to new CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def normalize_location(location):
    """Normalize location string by removing 'the' and extra whitespace"""
    return location.lower().replace('the', '').strip()

def generate_emoji_for_event(description):
    """
    Generates a relevant emoji for an event based on its description.
    """
    try:
        prompt = f"Based on this event description, suggest ONE single emoji that best represents it. Return ONLY the emoji, nothing else: '{description}'"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns exactly one emoji character based on the description provided. Return only the emoji, no text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.3
        )
        
        global TOTAL_TOKENS_USED
        TOTAL_TOKENS_USED += response.usage.total_tokens
        
        emoji = response.choices[0].message.content.strip()
        
        # Ensure we only return a single emoji
        if len(emoji) > 2:
            # If we got more than one emoji or text, just return a default
            return "🎉"
            
        return emoji
    except Exception as e:
        print(f"Error generating emoji: {str(e)}")
        return "🎉"  # Default emoji if generation fails

def create_newsletter_content(location_group, all_events):
    """Create newsletter content for a specific location group"""
    try:
        print(f"\nDEBUG: Location Group: {location_group}")
        
        # Filter events for this location group
        location_events = []
        for event in all_events:
            event_location = event['location'].lower().strip()
            event_group = get_location_group(event_location)
            if event_group == location_group:
                location_events.append(event)
        
        print(f"DEBUG: Found {len(location_events)} events for this location group")
        
        # Filter worth-the-drive events (different location group and marked with 'X')
        worth_drive_events = []
        for event in all_events:
            event_location = event['location'].lower().strip()
            event_group = get_location_group(event_location)
            if (event_group != location_group and
                event.get('worthTheDrive', '').strip().upper() == 'X'):
                worth_drive_events.append(event)
        
        # Skip if no events for this location group
        if not location_events:
            logger.info(f"No events found for {location_group}")
            return None
            
        # Process images and collect events
        event_sections = []
        worthTheDrive_sections = []
        
        # Process local events
        for event in location_events:
            # Handle image
            image_path = handle_image(event)
            
            # Generate emoji for the event
            emoji = generate_emoji_for_event(event.get('longDescription', event.get('description', event['activity'])))
            
            # Create event section HTML
            event_section = f'''
                <div>
                    {f"<img src='{image_path}'>" if image_path else '<p>[No Image Available]</p>'}
                    <h2><strong>{emoji} {event['daysOpen'].upper()} -- {event['activity']}</strong></h2>
                    <p>{event.get('longDescription', '')}</p>
                    <p><strong>💰  Cost:</strong> {event.get('cost', 'Not specified')}</p>
                    <p><strong>👶  Ages:</strong> {event.get('ages', 'Not specified')}</p>
                    <p><strong>📍  Address:</strong> {event.get('address', 'Not specified')}</p>
                    <p><strong>🕘  Hours:</strong> {event.get('hours', 'Not specified')}</p>
                    <p><strong>🔗  More Info: </strong> <a href='{event.get('url', '#')}'>{event.get('url', 'Not specified')}</a></p>
                    <h5>&nbsp;</h5>
                </div>
            '''
            event_sections.append(event_section)
            print(f"Added: {event['activity']} with emoji {emoji}")
        
        # Debug individual sections
        print("\nDEBUG: Individual event sections:")
        for i, section in enumerate(event_sections, 1):
            print(f"Event {i} length: {len(section)} characters")

        # Join sections and debug
        joined_sections = ''.join(event_sections)
        print(f"\nDEBUG: Joined sections length: {len(joined_sections)} characters")
        
        # Debug the final div
        events_div = f'''
        <div>
            {joined_sections}
        </div>
        '''
        print(f"DEBUG: Final events div length: {len(events_div)} characters")
        
        # Process worth-the-drive events
        for event in worth_drive_events:
            # Handle image
            image_path = handle_image(event)
            
            # Generate emoji for the event
            emoji = generate_emoji_for_event(event.get('longDescription', event.get('description', event['activity'])))
            
            # Create worth-the-drive section HTML
            worthTheDrive_section = f'''
                <div>
                    {f"<img src='{image_path}' style='max-width: 100%; height: auto; padding: 30px 30px 15px 30px; margin: 0 auto;'>" if image_path else '<p>[No Image Available]</p>'}
                    <h2><strong>{emoji} {event['daysOpen'].upper()} -- {event['activity']}</strong></h2>
                    <p>{event.get('longDescription', '')}</p>
                    <p><strong>💰  Cost:</strong> {event.get('cost', 'Not specified')}</p>
                    <p><strong>👶  Ages:</strong> {event.get('ages', 'Not specified')}</p>
                    <p><strong>📍  Address:</strong> {event.get('address', 'Not specified')}</p>
                    <p><strong>🕘  Hours:</strong> {event.get('hours', 'Not specified')}</p>
                    <p><strong>🔗  More Info: </strong> <a href='{event.get('url', '#')}'>{event.get('url', 'Not specified')}</a></p>
                    <h5>&nbsp;</h5>
                </div>
            '''
            worthTheDrive_sections.append(worthTheDrive_section)
         
        # Create the worth_drive_html variable before using it
        worth_drive_html = ""
        if worthTheDrive_sections:
            worth_drive_html = "<h1><strong>🚗  MIGHT BE WORTH THE DRIVE?  🚗</strong></h1>"
            worth_drive_html += "<p>Here's a few selected events from around LA, that might be worth the gas money.</p>"
            worth_drive_html += ''.join(worthTheDrive_sections)
        
        # Create complete HTML content
        html_content = f'''
            <div class="message-content">
                <h4>
                    {get_location_title(location_group)} Edition
                </h4>
                <h3>{get_upcoming_wednesday().strftime('%B %d, %Y')}</h3>

                <h6>
                    {shared_top_blurb}
                </h6>

                <!-- Main Events Section -->
                {events_div}
                
                <!-- Worth The Drive Section -->
                {worth_drive_html}
                
                <h6>
                    {shared_wrap_up}
                </h6>
                
            </div>
        '''
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error creating newsletter content for {location_group}: {e}")
        return None

def get_upcoming_wednesday():
    """Return the date of the upcoming Wednesday from today"""
    today = datetime.now().date()
    days_until_wednesday = (2 - today.weekday()) % 7  # Wednesday is weekday 2
    
    # If today is Wednesday, ask the user for a date
    if today.weekday() == 2:  # Wednesday is weekday 2
        print("\nToday is Wednesday. What date would you like to use for the newsletter?")
        print("Please enter in the format 'Month Day, Year' (e.g., 'June 15, 2024')")
        
        while True:
            date_input = input("> ").strip()
            try:
                # Parse the user input date
                user_date = datetime.strptime(date_input, "%B %d, %Y").date()
                return user_date
            except ValueError:
                print("Invalid date format. Please use the format 'Month Day, Year' (e.g., 'June 15, 2024')")
    
    # Otherwise calculate the upcoming Wednesday
    if days_until_wednesday == 0:  # If today is Wednesday, get next Wednesday
        days_until_wednesday = 7
    upcoming_wednesday = today + timedelta(days=days_until_wednesday)
    return upcoming_wednesday

# Main function
if __name__ == "__main__":
    try:
        # Copy original CSV to create update version
        shutil.copy2(INPUT_CSV, OUTPUT_CSV)
        
        # Get user input for content
        print("\nEnter the top blurb for the newsletter:")
        shared_top_blurb = input("> ").strip()
        shared_top_blurb += f"<br><br><em>- Steve & Matt</em>"  # Add signature with HTML formatting
        
        print("\nEnter the wrap-up message for the newsletter:")
        shared_wrap_up = input("> ").strip()
        
        print("\nGenerating newsletters...")
        all_events_to_update = []
        
        if 'convertkit' in globals():
            # Read events from CSV file
            with open(INPUT_CSV, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                all_events = list(reader)  # Read all events into a list
                all_events_to_update = []
                
                for location_group in LOCATIONS:
                    # Create and send newsletter for each location group
                    html_content = create_newsletter_content(location_group, all_events)
                    if html_content:
                        success = convertkit.send_newsletter(
                            content=html_content,
                            location=location_group,
                            preview=True
                        )
                        
                        if success:
                            logger.info(f"Newsletter draft created for {location_group}")
                        else:
                            logger.error(f"Failed to create newsletter for {location_group}")
                    
                    # Add any events that need updating to the main list
                    for event in all_events:
                        event_location = event['location'].lower().strip()
                        event_group = get_location_group(event_location)
                        if event_group == location_group:
                            all_events_to_update.append(event)

        # Update the CSV with new image paths if needed
        if all_events_to_update:
            create_updated_csv(all_events_to_update, INPUT_CSV, OUTPUT_CSV)
            print(f"\nUpdated CSV saved to {OUTPUT_CSV}")
            
        print(f"\nTotal tokens used: {TOTAL_TOKENS_USED}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")