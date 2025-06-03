from docx import Document
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import requests
from io import BytesIO
import subprocess
import sys
import inquirer
import time
from datetime import datetime, date
import csv
import logging
import webbrowser
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths and settings
FONT_PATH = "TiltWarp.ttf"  # Replace with your TTF font file path
OUTPUT_FOLDER = "output/"
EVENTS_CSV = "events.csv"  # Changed from data.csv
ASSETS_FOLDER = "assets/"  # Add assets folder for DALL-E images
NETLIFY_BASE_URL = os.getenv("NETLIFY_BASE_URL", "https://jocular-peony-417429.netlify.app/output/")  # Note the /output/ path

# Ensure output and assets folders exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ASSETS_FOLDER, exist_ok=True)

def run_newsletter_writer():
    """Run the newsletter-writer.py script"""
    try:
        print("\n🚀 Running newsletter writer...")
        result = subprocess.run([sys.executable, 'newsletter-writer.py'], 
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              check=True)
        print("✅ Newsletter writer completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running newsletter writer: {e}")
        return False
    except FileNotFoundError:
        print("❌ newsletter-writer.py not found in the current directory")
        return False

def push_to_netlify(max_retries=3, retry_delay=5):
    """Push new images to Netlify via Git with confirmation and retry mechanism"""
    
    # Check if there are any changes to push
    try:
        # Get all untracked and modified files
        status_result = subprocess.run(['git', 'status', '--porcelain'], 
                                     capture_output=True, text=True, check=True)
        
        # Filter for files in output/ and assets/ folders
        relevant_files = []
        for line in status_result.stdout.strip().split('\n'):
            if line.strip():  # Skip empty lines
                # Extract filename (after the status code)
                filename = line[3:].strip()
                if (filename.startswith('output/') and 
                    (filename.endswith('.png') or filename.endswith('.docx') or filename.endswith('.csv'))) or \
                   filename.startswith('assets/'):
                    relevant_files.append(line.strip())
        
        if not relevant_files:
            print("No new files to push to Netlify")
            return False  # Changed from True to False to indicate no push occurred
            
        # Show what files will be pushed
        print("\n" + "="*60)
        print("NETLIFY PUSH PREPARATION")
        print("="*60)
        print("The following files will be pushed to Netlify:")
        for file_line in relevant_files:
            print(f"  {file_line}")
        print("="*60)
        
    except subprocess.CalledProcessError as e:
        print(f"Error checking git status: {e}")
        return False

    # Ask for confirmation
    questions = [
        inquirer.Confirm('push_to_netlify',
                        message="Would you like to push the new files to Netlify?",
                        default=True),
    ]
    
    if not inquirer.prompt(questions)['push_to_netlify']:
        print("Skipping Netlify push. Remember to push the files manually!")
        return False

    # Additional confirmation before actual push
    print("\n⚠️  FINAL CONFIRMATION ⚠️")
    print("This will commit and push files to the live Netlify site.")
    
    final_confirmation = input("Type 'PUSH' to continue or anything else to cancel: ").strip()
    
    if final_confirmation != 'PUSH':
        print("Push cancelled by user.")
        return False

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1} of {max_retries}...")
                time.sleep(retry_delay)

            # Add all new files in the output and assets folders
            print("Adding new files to Git...")
            subprocess.run(['git', 'add', 'output/', 'assets/'], check=True)
            
            # Show what's being committed
            print("\nFiles to be committed:")
            subprocess.run(['git', 'status'], check=True)
            
            # Commit the changes
            print("\nCommitting changes...")
            subprocess.run(['git', 'commit', '-m', 'Update social media content and assets'], check=True)
            
            # Push to main branch
            print("\nPushing to Netlify...")
            subprocess.run(['git', 'push', 'origin', 'main'], check=True)
            
            print("\n✅ Successfully pushed new files to Netlify!")
            
            # Ask if user wants to run newsletter writer
            newsletter_questions = [
                inquirer.Confirm('run_newsletter',
                                message="Would you like to run the newsletter writer now?",
                                default=True),
            ]
            
            if inquirer.prompt(newsletter_questions)['run_newsletter']:
                return run_newsletter_writer()
            else:
                print("Skipping newsletter writer.")
                return True

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error during Git operation: {e}")
            if attempt < max_retries - 1:
                print(f"\nWill retry in {retry_delay} seconds...")
            else:
                print("\n❌ Failed to push to Netlify after all retry attempts")
                return False

    return False

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
                
                print(f"Saved DALL-E image to: {local_path}")
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

def fetch_image_from_url(url):
    """Fetch an image from a URL and return it as a PIL Image object."""
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert("RGBA")
    else:
        raise ValueError(f"Failed to fetch image from URL: {url}")


def crop_and_resize(image, size):
    """Crop the center of the image and resize to match the template size."""
    img_width, img_height = image.size
    target_width, target_height = size

    aspect_ratio_img = img_width / img_height
    aspect_ratio_target = target_width / target_height

    if aspect_ratio_img > aspect_ratio_target:
        # Crop width
        new_width = int(img_height * aspect_ratio_target)
        left = (img_width - new_width) // 2
        box = (left, 0, left + new_width, img_height)
    else:
        # Crop height
        new_height = int(img_width / aspect_ratio_target)
        top = (img_height - new_height) // 2
        box = (0, top, img_width, top + new_height)

    cropped_image = image.crop(box)
    return cropped_image.resize((target_width, target_height), Image.LANCZOS)


def create_caption(row):
    """Create a caption for the given row."""
    caption = (
        f"{row['longDescription']}\n\n"
        f"📍 Location: {row['facility']}\n"
        f"{row['address']}\n"
        f"📅 Date: {row['dates']}\n"
        f"🕘 Time: {row['hours']}\n"
        f"💰 Tickets: {row['cost']}\n"
        f"👶 Age Requirement: {row['ages']}\n"
        f"🔗 More info: {row['url']}\n\n"
        f"{row.get('hashtags', '')} #shittodowithkids #stdwkids #familyactivities #kidslosangeles"
    )
    return caption

def sanitize_filename(filename):
    """Sanitize filename to remove problematic characters"""
    # Replace spaces and special characters with underscores
    sanitized = filename.replace(' ', '_')
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '_-')
    return sanitized

def create_image(row):
    """Create the image and its caption."""
    try:
        # Settings for location-based template and text color
        location_settings = {
            "downtown": {"template_name": "downtown.png", "color": "#F7931E"},
            "eastla": {"template_name": "EastLA.png", "color": "#662D91"},
            "hollywood": {"template_name": "Hollywood.png", "color": "#ED1E79"},
            "longbeach": {"template_name": "LongBeach.png", "color": "#E0CE20"},
            "midcity": {"template_name": "MidCity.png", "color": "#00E0B5"},
            "southbay": {"template_name": "SouthBay.png", "color": "#FF7BAC"},
            "southla": {"template_name": "SouthLA.png", "color": "#ED1C24"},
            "thevalley": {"template_name": "TheValley.png", "color": "#9E005D"},
            "westside": {"template_name": "Westside.png", "color": "#C7B299"},
            "westvalley": {"template_name": "WestValley.png", "color": "#00FF4A"},
            "outsidela": {"template_name": "outsideLA.png", "color": "#2E3192"},
        }

        location = row["location"].lower()
        settings = location_settings.get(location)
        if not settings:
            raise ValueError(f"No template or settings found for location '{location}'")

        # Check for local template first
        template_name = settings["template_name"]
        local_template_path = os.path.join("overlays", template_name)
        if os.path.exists(local_template_path):
            print(f"Using local template for location '{location}'")
            template = Image.open(local_template_path).convert("RGBA")
        else:
            print(f"Local template not found for '{location}', fetching from URL.")
            remote_template_url = f"https://shittodowithkids.com/social_overlays/{template_name}"
            template = fetch_image_from_url(remote_template_url)

        template_size = template.size
        neighborhood_color = settings["color"]

        # Fetch the background image
        generated_image_path = None
        if not pd.notna(row["imageUrl"]) or str(row["imageUrl"]).strip() == "":
            # No image URL provided, generate with DALL-E
            print(f"No image URL for {row['activity']}, generating with DALL-E...")
            generated_image_path = generate_dalle_image(row.get('description', row['activity']), row['activity'])
            if generated_image_path:
                # Update the row with the new image path
                row = row.copy()
                row["imageUrl"] = generated_image_path
                row["new_image_path"] = generated_image_path
                print(f"Using DALL-E generated image: {generated_image_path}")
            else:
                raise ValueError(f"Failed to generate DALL-E image for {row['activity']}")
                
        if pd.notna(row["imageUrl"]) and row["imageUrl"].startswith("http"):
            background = fetch_image_from_url(row["imageUrl"])
        else:
            # Use a local image if the URL is not valid
            local_image_path = os.path.join("assets", os.path.basename(row["imageUrl"]))
            if not os.path.exists(local_image_path):
                raise ValueError(f"Local image not found: {local_image_path}")
            background = Image.open(local_image_path).convert("RGBA")
        
        # Resize and crop the background
        background = crop_and_resize(background, template_size)

        # Overlay the template on the background
        background.paste(template, (0, 0), template)

        # Create a drawing context
        draw = ImageDraw.Draw(background)

        # Fonts and text adjustments
        font_title_size = 50
        font_description_size = 30
        font_title = ImageFont.truetype(FONT_PATH, size=font_title_size)
        font_description = ImageFont.truetype(FONT_PATH, size=font_description_size)

        # Add "activity" text
        activity_text = row["activity"].upper()
        max_width = 640
        y_offset = 1010

        # Shrink font size until text fits within max_width
        while True:
            text_bbox = draw.textbbox((0, 0), activity_text, font=font_title)
            text_width = text_bbox[2] - text_bbox[0]
            if text_width <= max_width or font_title_size <= 15:
                break
            font_title_size -= 1
            font_title = ImageFont.truetype(FONT_PATH, size=font_title_size)

        draw.text((390, y_offset), activity_text, font=font_title, fill="white")
        y_offset += font_title_size + 25

        # Add a horizontal line
        line_y = y_offset - 5
        draw.line([(390, line_y), (390 + 640, line_y)], fill="white", width=2)

        # Add "neighborhood" text
        neighborhood_text = row["neighborhood"].upper()
        draw.text((390, y_offset), neighborhood_text, font=font_description, fill=neighborhood_color)
        y_offset += font_description_size + 10

        # Add wrapped "description" text
        description_text = row["description"].upper()
        wrapped_text = ""
        words = description_text.split()
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            test_bbox = draw.textbbox((0, 0), test_line, font=font_description)
            if test_bbox[2] - test_bbox[0] <= max_width:
                current_line = test_line
            else:
                wrapped_text += f"{current_line}\n"
                current_line = word
        wrapped_text += current_line

        for line in wrapped_text.split("\n"):
            draw.text((390, y_offset), line, font=font_description, fill="white")
            y_offset += font_description_size + 5

        # Save the final image with thoroughly sanitized filename
        filename = sanitize_filename(row['activity'])
        image_path = os.path.join(OUTPUT_FOLDER, f"{filename}.png")
        background.save(image_path, "PNG")
        print(f"Created: {image_path}")

        # Create individual caption document with matching filename
        caption_doc = Document()
        caption_doc.add_paragraph(create_caption(row))
        caption_path = os.path.join(OUTPUT_FOLDER, f"{filename}_caption.docx")
        caption_doc.save(caption_path)
        print(f"Created caption: {caption_path}")
        
        # Return the updated row if we generated a new image
        if generated_image_path:
            return row
        return None

    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def format_date(date_str):
    """Convert various date formats to YYYY-MM-DD"""
    try:
        # Normalize dashes - replace en-dash with regular hyphen
        date_str = date_str.replace('–', '-')
        
        # Handle range dates (e.g., "Feb 14-16, 2025")
        if '-' in date_str:
            # Take the first date of the range
            start_date = date_str.split('-')[0].strip()
            if ',' in date_str:  # Has year in it
                year = date_str.split(',')[1].strip()
                start_date = f"{start_date}, {year}"
            try:
                # Parse the start date
                date_obj = datetime.strptime(start_date, '%b %d, %Y')
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                # Try alternate format
                date_obj = datetime.strptime(start_date, '%B %d, %Y')
                return date_obj.strftime('%Y-%m-%d')
        
        # Handle single dates
        if ',' in date_str:  # Already in Month Day, Year format
            try:
                date_obj = datetime.strptime(date_str.strip(), '%b %d, %Y')
            except ValueError:
                date_obj = datetime.strptime(date_str.strip(), '%B %d, %Y')
            return date_obj.strftime('%Y-%m-%d')
            
        # If already in YYYY-MM-DD format
        if date_str.count('-') == 2:
            return date_str.strip()
            
    except Exception as e:
        print(f"Error formatting date {date_str}: {e}")
        return date_str  # Return original if parsing fails

def create_scheduler_csv(data_rows):
    """Create InstagramSchedulerPrep.csv with sorted events"""
    csv_path = 'InstagramSchedulerPrep.csv'
    
    try:
        # Prepare data for CSV
        scheduler_data = []
        for row in data_rows:
            scheduler_entry = {
                'eventDate': format_date(row['dates']),
                'socialPost': f"{NETLIFY_BASE_URL}{sanitize_filename(row['activity'])}.png",
                'socialCaption': create_caption(row)
            }
            scheduler_data.append(scheduler_entry)

        # Convert to DataFrame for easy sorting
        df = pd.DataFrame(scheduler_data)
        
        # Sort by eventDate (already in YYYY-MM-DD format)
        df = df.sort_values('eventDate')
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Created scheduler CSV: {csv_path}")
        
    except Exception as e:
        print(f"Error creating scheduler CSV: {e}")
        raise

def create_updated_events_csv(data_rows):
    """Create events_update.csv with updated imageUrl values pointing to Netlify"""
    csv_path = 'events_update.csv'  # Save to root folder instead of OUTPUT_FOLDER
    
    try:
        # Create a deep copy of the data
        updated_data = []
        for row in data_rows:
            # Create a copy of the row
            updated_row = row.copy()
            
            # Update the imageUrl to point to the Netlify URL
            activity_filename = sanitize_filename(row['activity'])
            netlify_url = f"{NETLIFY_BASE_URL}{activity_filename}.png"
            updated_row['imageUrl'] = netlify_url
            
            updated_data.append(updated_row)

        # Convert to DataFrame
        df = pd.DataFrame(updated_data)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Created updated events CSV: {csv_path}")
        
    except Exception as e:
        print(f"Error creating updated events CSV: {e}")
        raise

# Add near the top of your script
def print_problematic_rows(data):
    print("\n--- ROWS WITH MISSING VALUES ---")
    
    # Check for NaN in critical columns
    missing_location = data[data['location'].isna()]
    missing_activity = data[data['activity'].isna()]
    missing_dates = data[data['dates'].isna()]
    
    if not missing_location.empty:
        print(f"\nMissing LOCATION ({len(missing_location)} rows):")
        print(missing_location[['activity', 'location']])
    
    if not missing_activity.empty:
        print(f"\nMissing ACTIVITY ({len(missing_activity)} rows):")
        print(missing_activity[['activity', 'location']])
    
    if not missing_dates.empty:
        print(f"\nMissing DATES ({len(missing_dates)} rows):")
        print(missing_dates[['activity', 'dates']])
    
    # Check for special characters in dates
    has_endash = data[data['dates'].notna() & data['dates'].str.contains('–', na=False)]
    if not has_endash.empty:
        print(f"\nDates with en-dash ({len(has_endash)} rows):")
        print(has_endash[['activity', 'dates']])

def clear_output_folder():
    """Remove all files from the output folder before starting a new run"""
    try:
        print(f"Clearing previous files from {OUTPUT_FOLDER}...")
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"  Removed: {filename}")
        print("Output folder cleared successfully")
    except Exception as e:
        print(f"Error clearing output folder: {e}")

# Update main processing loop
def main():
    try:
        # Clear the output folder before starting
        clear_output_folder()
        
        # Read the CSV file
        data = pd.read_csv(EVENTS_CSV)
        
        # Print column names to debug
        print("Available columns in CSV:", data.columns.tolist())
        
        # Check for required columns
        required_columns = ['location', 'activity', 'neighborhood', 'description', 
                          'longDescription', 'facility', 'address', 'dates', 
                          'hours', 'cost', 'ages', 'url', 'imageUrl']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # Call this function in main() right after reading the CSV
        print_problematic_rows(data)
        
        data_rows = data.to_dict('records')
        updated_rows = []
        
        # Process each row
        for row in data_rows:
            try:
                updated_row = create_image(row)
                if updated_row:
                    # If we got an updated row (with a new DALL-E image), use it
                    updated_rows.append(updated_row)
                else:
                    # Otherwise use the original row
                    updated_rows.append(row)
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                print(f"Row data: {row}")  # Print the problematic row
                # Still add the original row to keep all events
                updated_rows.append(row)
        
        # Create scheduler CSV
        create_scheduler_csv(updated_rows)
        
        # Create updated events CSV with Netlify URLs
        create_updated_events_csv(updated_rows)
        
        # Push to Netlify (and potentially run newsletter writer)
        push_result = push_to_netlify()
        if push_result:
            print("\n🎉 Process completed successfully!")
            print("Files have been pushed to Netlify and/or newsletter writer was run")
            print(f"Scheduler CSV created in InstagramSchedulerPrep.csv")
            print(f"Updated events CSV created in events_update.csv")
        else:
            print("\n⚠️  Process completed but files were not pushed to Netlify")
            print("Files are ready for manual push if needed")
            print(f"Scheduler CSV created in InstagramSchedulerPrep.csv")
            print(f"Updated events CSV created in events_update.csv")
            
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()