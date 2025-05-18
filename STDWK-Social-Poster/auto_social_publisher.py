# Defunct script for syncing social posts directly to Instagram.  Unable to work due to API restrictions for non-app-based access.

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import pytz
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.igmedia import IGMedia
from facebook_business.adobjects.page import Page
from docx import Document
import inquirer
from facebook_business.adobjects.iguser import IGUser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('social_publisher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Instagram API Configuration
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")

# Initialize Facebook API
FacebookAdsApi.init(access_token=ACCESS_TOKEN)

class InstagramPublisher:
    def __init__(self):
        """Initialize the Instagram Publisher with necessary configurations"""
        self.api = FacebookAdsApi.init(access_token=ACCESS_TOKEN)
        self.ig_user = IGUser(INSTAGRAM_ACCOUNT_ID)
        self.output_folder = "output"
        self.posted_log = "posted_content.json"
        self.timezone = pytz.timezone('America/Los_Angeles')
        self.netlify_base_url = os.getenv("NETLIFY_BASE_URL", "https://jocular-peony-417429.netlify.app/output/")
        self.scheduled_posts = []  # Track scheduled posts
        self.failed_posts = []     # Track failed posts
        self.load_posted_content()

    def load_posted_content(self):
        """Load the log of previously posted content"""
        try:
            if os.path.exists(self.posted_log):
                with open(self.posted_log, 'r') as f:
                    content = f.read().strip()
                    if content:  # Check if file has content
                        self.posted_content = json.load(f)
                    else:
                        logger.info("Posted content log is empty")
                        self.posted_content = []
            else:
                logger.info("No posted content log found")
                self.posted_content = []
        except Exception as e:
            logger.error(f"Error loading posted content log: {str(e)}")
            self.posted_content = []

    def save_posted_content(self):
        """Save the log of posted content"""
        try:
            with open(self.posted_log, 'w') as f:
                json.dump(self.posted_content, f)
        except Exception as e:
            logger.error(f"Error saving posted content log: {str(e)}")

    def get_unposted_content(self):
        """Get list of unposted content from the output folder"""
        try:
            # Get all files from output folder
            all_files = os.listdir(self.output_folder)
            logger.info(f"\nAll files in output folder: {all_files}")
            
            # Get PNG files
            image_files = [f for f in all_files if f.endswith('.png')]
            logger.info(f"PNG files found: {image_files}")
            
            # Get previously posted files
            posted_files = [p['filename'] for p in self.posted_content]
            logger.info(f"Previously posted files: {posted_files}")
            
            # Get unposted files
            unposted = [f for f in image_files if f not in posted_files]
            logger.info(f"Unposted files: {unposted}")
            
            return unposted
        except Exception as e:
            logger.error(f"Error getting unposted content: {str(e)}")
            return []

    def get_caption_for_image(self, image_filename):
        """Get the corresponding caption from the individual caption file"""
        try:
            # Get base name without extension (preserving case)
            base_name = os.path.splitext(image_filename)[0]
            caption_file = os.path.join(self.output_folder, f"{base_name}_caption.docx")
            
            logger.info(f"Looking for caption file: {caption_file}")
            
            if not os.path.exists(caption_file):
                # Try case-sensitive match
                logger.error(f"Caption file not found: {caption_file}")
                return "No caption available"

            # Read the caption from the Word document
            doc = Document(caption_file)
            caption = '\n'.join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
            
            if caption:
                logger.info(f"Found caption for {image_filename}")
                return caption
            else:
                logger.error(f"Empty caption file for {image_filename}")
                return "No caption available"

        except Exception as e:
            logger.error(f"Error getting caption for {image_filename}: {str(e)}")
            return "Error getting caption"

    def get_next_monday(self):
        """Calculate the date of the next Monday"""
        today = datetime.now(self.timezone)
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0 and today.hour >= 19:  # If it's Monday after 7pm
            days_until_monday = 7  # Go to next Monday
        next_monday = today + timedelta(days=days_until_monday)
        return next_monday

    def schedule_post_to_instagram(self, image_filename, scheduled_time):
        """Schedule a post to Instagram using Meta's API"""
        try:
            # Convert local time to UTC timestamp
            utc_timestamp = int(scheduled_time.astimezone(pytz.UTC).timestamp())
            current_timestamp = int(datetime.now(pytz.UTC).timestamp())
            
            # Validate scheduling time is in the future
            if utc_timestamp <= current_timestamp:
                logger.error(f"Scheduled time {scheduled_time} is not in the future!")
                self.failed_posts.append({
                    'filename': image_filename,
                    'scheduled_for': scheduled_time.isoformat(),
                    'error': 'Scheduled time must be in the future'
                })
                return False
            
            # Ensure minimum 1 hour in the future
            if utc_timestamp < (current_timestamp + 3600):
                logger.error(f"Scheduled time must be at least 1 hour in the future")
                self.failed_posts.append({
                    'filename': image_filename,
                    'scheduled_for': scheduled_time.isoformat(),
                    'error': 'Schedule time must be at least 1 hour ahead'
                })
                return False

            # Get caption (using cache)
            if not hasattr(self, '_caption_cache'):
                self._caption_cache = {}
            
            if image_filename not in self._caption_cache:
                caption = self.get_caption_for_image(image_filename)
                self._caption_cache[image_filename] = caption
            else:
                caption = self._caption_cache[image_filename]

            image_url = self.netlify_base_url + image_filename
            logger.info(f"Scheduling post for {scheduled_time} (UTC timestamp: {utc_timestamp})")
            logger.info(f"Current time UTC: {datetime.fromtimestamp(current_timestamp, pytz.UTC)}")
            logger.info(f"Image URL: {image_url}")

            # Create media container
            container = self.ig_user.create_media(
                params={
                    'image_url': image_url,
                    'caption': caption,
                    'published': False,
                    'scheduled_publish_time': utc_timestamp  # Add timestamp here too
                }
            )
            
            if not container or 'id' not in container:
                logger.error("Failed to create media container")
                self.failed_posts.append({
                    'filename': image_filename,
                    'scheduled_for': scheduled_time.isoformat(),
                    'error': 'Media container creation failed'
                })
                return False

            # Schedule the post
            result = self.ig_user.create_media_publish(
                params={
                    'creation_id': container['id'],
                    'scheduled_publish_time': utc_timestamp
                }
            )

            # Validate scheduling response
            if result and 'id' in result:
                logger.info(f"Successfully scheduled post ID {result['id']} for {scheduled_time}")
                # Verify scheduling
                try:
                    # Add verification of scheduling status if API provides it
                    logger.info(f"Full scheduling response: {result}")
                except Exception as e:
                    logger.warning(f"Could not verify scheduling status: {e}")
                
                self.scheduled_posts.append({
                    'filename': image_filename,
                    'scheduled_for': scheduled_time.isoformat(),
                    'post_id': result['id']
                })
                return True
            else:
                logger.error(f"Failed to schedule post: {result}")
                self.failed_posts.append({
                    'filename': image_filename,
                    'scheduled_for': scheduled_time.isoformat(),
                    'error': 'Scheduling failed'
                })
                return False

        except Exception as e:
            logger.error(f"Error scheduling post: {str(e)}")
            self.failed_posts.append({
                'filename': image_filename,
                'scheduled_for': scheduled_time.isoformat(),
                'error': str(e)
            })
            return False

    def schedule_posts(self):
        """Schedule posts for 5am and 7pm Monday through Friday"""
        start_date = self.get_next_monday()
        unposted = self.get_unposted_content()
        posts_needed_per_week = 10  # 2 posts per day, 5 days per week
        
        logger.info(f"\nScheduling Overview:")
        logger.info(f"Start Date: {start_date}")
        logger.info(f"Available Posts: {len(unposted)}")
        logger.info(f"Posts Needed Per Week: {posts_needed_per_week}")
        logger.info(f"Weeks of Content: {len(unposted) / posts_needed_per_week:.1f}")

        if not unposted:
            logger.info("No unposted content found.")
            return False

        if len(unposted) < posts_needed_per_week:
            questions = [
                inquirer.Confirm('proceed_anyway',
                               message=f"Only {len(unposted)} posts available (need {posts_needed_per_week} for a full week). Proceed?",
                               default=False),
            ]
            if not inquirer.prompt(questions)['proceed_anyway']:
                logger.info("Scheduling cancelled due to insufficient content.")
                return False

        # Schedule confirmation
        questions = [
            inquirer.Confirm('confirm_schedule',
                           message="Proceed with scheduling these posts?",
                           default=False),
        ]
        if not inquirer.prompt(questions)['confirm_schedule']:
            logger.info("Scheduling cancelled by user.")
            return False

        # Clear tracking lists
        self.scheduled_posts = []
        self.failed_posts = []
        
        # Schedule posts
        current_date = start_date
        max_attempts = len(unposted) * 2  # Safety limit
        attempt = 0
        
        for current_index in range(len(unposted)):
            logger.info(f"\nProcessing post {current_index + 1} of {len(unposted)}")
            
            if attempt >= max_attempts:
                logger.error("Maximum scheduling attempts reached. Stopping.")
                break
            
            attempt += 1
            
            try:
                # Schedule the post
                scheduled_time = current_date.replace(
                    hour=5 if current_index % 2 == 0 else 19,
                    minute=0, second=0, microsecond=0
                )
                
                logger.info(f"Attempting to schedule {unposted[current_index]} for {scheduled_time}")
                
                if self.schedule_post_to_instagram(unposted[current_index], scheduled_time):
                    self.posted_content.append(self.scheduled_posts[-1])
                    self.save_posted_content()
                    logger.info(f"Successfully scheduled post {current_index + 1}")
                    
                    # Move to next time slot
                    if current_index % 2 == 1:  # After evening post
                        current_date += timedelta(days=1)
                        while current_date.weekday() >= 5:  # Skip weekend days
                            current_date += timedelta(days=1)
                else:
                    logger.error(f"Failed to schedule post {current_index + 1}")
                
            except Exception as e:
                logger.error(f"Error scheduling post {current_index + 1}: {str(e)}")
                continue

        # Final status report
        logger.info("\nScheduling Complete!")
        logger.info(f"Successfully scheduled: {len(self.scheduled_posts)} posts")
        if self.failed_posts:
            logger.error(f"Failed to schedule: {len(self.failed_posts)} posts")
            for failed in self.failed_posts:
                logger.error(f"Failed post: {failed['filename']} - {failed['error']}")
            return False
        
        return len(self.scheduled_posts) > 0

def main():
    """Main function to run the publisher"""
    try:
        questions = [
            inquirer.Confirm('run_publisher',
                           message="Do you want to schedule social posts?",
                           default=False),
        ]
        answers = inquirer.prompt(questions)

        if not answers['run_publisher']:
            logger.info("Social publisher not started. Exiting.")
            return

        publisher = InstagramPublisher()
        if publisher.schedule_posts():
            logger.info("All posts have been scheduled successfully!")
        else:
            logger.info("Post scheduling cancelled.")

    except KeyboardInterrupt:
        logger.info("Social publisher stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
