#!/usr/bin/env python3
"""
Multi-Platform Video Uploader
Upload videos to YouTube, TikTok, and Twitter/X with a single script.

Author: Assistant
Dependencies: google-api-python-client, google-auth-oauthlib, tweepy, TikTokApi, tiktokautouploader
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
from pathlib import Path

# YouTube imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# Twitter/X imports
import tweepy

# TikTok imports (using automation approach due to API limitations)
try:
    from tiktokautouploader import upload_tiktok
    TIKTOK_AVAILABLE = True
except ImportError:
    TIKTOK_AVAILABLE = False
    print("Warning: tiktokautouploader not installed. TikTok upload will be disabled.")

# Configuration
PLATFORMS = ['youtube', 'tiktok', 'twitter']
MAX_VIDEO_SIZE = 8 * 1024 * 1024 * 1024  # 8GB limit for most platforms

# YouTube API configuration
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Video categories for YouTube
YOUTUBE_CATEGORIES = {
    'entertainment': '24',
    'education': '27', 
    'howto': '26',
    'music': '10',
    'news': '25',
    'nonprofit': '29',
    'people': '22',  # People & Blogs
    'pets': '15',
    'sports': '17',
    'tech': '28',
    'travel': '19',
    'gaming': '20'
}

class MultiPlatformUploader:
    """Main class for uploading videos to multiple platforms."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config_dir / 'upload.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Platform clients
        self.youtube_service = None
        self.twitter_client = None
        
    def authenticate_youtube(self) -> bool:
        """Authenticate with YouTube API."""
        try:
            creds = None
            token_file = self.config_dir / 'youtube_token.json'
            credentials_file = self.config_dir / 'youtube_credentials.json'
            
            if not credentials_file.exists():
                self.logger.error(f"YouTube credentials file not found: {credentials_file}")
                self.logger.info("Please download your OAuth2 credentials from Google Cloud Console")
                self.logger.info("and save as 'config/youtube_credentials.json'")
                return False
            
            # Load existing token
            if token_file.exists():
                creds = Credentials.from_authorized_user_file(str(token_file), YOUTUBE_SCOPES)
            
            # If no valid credentials, run OAuth flow
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_file), YOUTUBE_SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save credentials for next run
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
            
            self.youtube_service = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=creds)
            self.logger.info("YouTube authentication successful")
            return True
            
        except Exception as e:
            self.logger.error(f"YouTube authentication failed: {e}")
            return False
    
    def authenticate_twitter(self) -> bool:
        """Authenticate with Twitter API."""
        try:
            # Load Twitter credentials from environment or config
            api_key = os.getenv('TWITTER_API_KEY')
            api_secret = os.getenv('TWITTER_API_SECRET')
            access_token = os.getenv('TWITTER_ACCESS_TOKEN')
            access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            
            if not all([api_key, api_secret, access_token, access_token_secret, bearer_token]):
                self.logger.error("Twitter credentials not found in environment variables")
                self.logger.info("Please set: TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET, TWITTER_BEARER_TOKEN")
                return False
            
            # Create Twitter client for API v2
            self.twitter_client = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                wait_on_rate_limit=True
            )
            
            # Create API v1.1 client for media upload
            auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
            self.twitter_api_v1 = tweepy.API(auth)
            
            self.logger.info("Twitter authentication successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Twitter authentication failed: {e}")
            return False
    
    def upload_to_youtube(self, video_path: str, title: str, description: str, 
                         category: str = 'people', privacy: str = 'public',
                         tags: Optional[List[str]] = None) -> Dict:
        """Upload video to YouTube."""
        try:
            if not self.youtube_service:
                if not self.authenticate_youtube():
                    return {'success': False, 'error': 'YouTube authentication failed'}
            
            category_id = YOUTUBE_CATEGORIES.get(category.lower(), '22')
            
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags or [],
                    'categoryId': category_id
                },
                'status': {
                    'privacyStatus': privacy
                }
            }
            
            media_body = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            request = self.youtube_service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media_body
            )
            
            response = None
            error = None
            retry = 0
            
            while response is None:
                try:
                    self.logger.info("Uploading to YouTube...")
                    status, response = request.next_chunk()
                    if status:
                        self.logger.info(f"Upload progress: {int(status.progress() * 100)}%")
                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504]:
                        error = f"A retriable HTTP error {e.resp.status} occurred"
                        self.logger.warning(error)
                    else:
                        raise
                except Exception as e:
                    error = f"A retriable error occurred: {e}"
                    self.logger.warning(error)
                
                if error is not None:
                    retry += 1
                    if retry > 3:
                        return {'success': False, 'error': 'Max retries exceeded'}
                    
                    import time
                    import random
                    sleep_seconds = random.random() * (2 ** retry)
                    self.logger.info(f"Retrying in {sleep_seconds:.2f} seconds...")
                    time.sleep(sleep_seconds)
            
            if 'id' in response:
                video_url = f"https://youtube.com/watch?v={response['id']}"
                self.logger.info(f"YouTube upload successful: {video_url}")
                return {
                    'success': True, 
                    'video_id': response['id'], 
                    'url': video_url
                }
            else:
                return {'success': False, 'error': f'Unexpected response: {response}'}
            
        except Exception as e:
            self.logger.error(f"YouTube upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def upload_to_twitter(self, video_path: str, title: str, description: str) -> Dict:
        """Upload video to Twitter/X."""
        try:
            if not self.twitter_client:
                if not self.authenticate_twitter():
                    return {'success': False, 'error': 'Twitter authentication failed'}
            
            # Check file size (Twitter has 512MB limit)
            file_size = os.path.getsize(video_path)
            if file_size > 512 * 1024 * 1024:  # 512MB
                return {'success': False, 'error': 'Video file too large for Twitter (max 512MB)'}
            
            # Upload media using v1.1 API
            self.logger.info("Uploading video to Twitter...")
            media = self.twitter_api_v1.media_upload(video_path)
            
            # Create tweet with media using v2 API
            tweet_text = f"{title}\n\n{description}"
            if len(tweet_text) > 280:
                tweet_text = tweet_text[:277] + "..."
            
            response = self.twitter_client.create_tweet(
                text=tweet_text,
                media_ids=[media.media_id]
            )
            
            tweet_url = f"https://twitter.com/user/status/{response.data['id']}"
            self.logger.info(f"Twitter upload successful: {tweet_url}")
            return {
                'success': True,
                'tweet_id': response.data['id'],
                'url': tweet_url
            }
            
        except Exception as e:
            self.logger.error(f"Twitter upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def upload_to_tiktok(self, video_path: str, title: str, description: str,
                        account_name: str = 'default') -> Dict:
        """Upload video to TikTok using automation."""
        try:
            if not TIKTOK_AVAILABLE:
                return {'success': False, 'error': 'TikTok uploader not available'}
            
            # Check file size (TikTok has specific limits)
            file_size = os.path.getsize(video_path)
            if file_size > 287 * 1024 * 1024:  # ~287MB is typical TikTok limit
                return {'success': False, 'error': 'Video file too large for TikTok'}
            
            # Combine title and description for TikTok
            tiktok_description = f"{title}\n\n{description}"
            if len(tiktok_description) > 300:  # TikTok caption limit
                tiktok_description = tiktok_description[:297] + "..."
            
            self.logger.info("Uploading to TikTok...")
            
            # Use tiktokautouploader
            upload_tiktok(
                video=video_path,
                description=tiktok_description,
                accountname=account_name
            )
            
            self.logger.info("TikTok upload initiated (check TikTok app for completion)")
            return {
                'success': True,
                'message': 'TikTok upload initiated - check the app for completion status'
            }
            
        except Exception as e:
            self.logger.error(f"TikTok upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def upload_to_platforms(self, video_path: str, title: str, description: str,
                           platforms: List[str], **kwargs) -> Dict:
        """Upload video to specified platforms."""
        if not os.path.exists(video_path):
            return {'error': f'Video file not found: {video_path}'}
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size > MAX_VIDEO_SIZE:
            return {'error': f'Video file too large: {file_size / (1024**3):.2f}GB'}
        
        results = {}
        
        for platform in platforms:
            platform = platform.lower()
            
            if platform == 'youtube':
                results['youtube'] = self.upload_to_youtube(
                    video_path, title, description,
                    category=kwargs.get('youtube_category', 'people'),
                    privacy=kwargs.get('youtube_privacy', 'public'),
                    tags=kwargs.get('youtube_tags')
                )
            
            elif platform == 'twitter':
                results['twitter'] = self.upload_to_twitter(video_path, title, description)
            
            elif platform == 'tiktok':
                results['tiktok'] = self.upload_to_tiktok(
                    video_path, title, description,
                    account_name=kwargs.get('tiktok_account', 'default')
                )
            
            else:
                results[platform] = {'success': False, 'error': f'Unsupported platform: {platform}'}
        
        return results


def create_sample_config():
    """Create sample configuration files."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Sample environment file
    env_sample = """# Twitter/X API Configuration
TWITTER_API_KEY=your_api_key_here
TWITTER_API_SECRET=your_api_secret_here
TWITTER_ACCESS_TOKEN=your_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret_here
TWITTER_BEARER_TOKEN=your_bearer_token_here

# Optional: TikTok account name
TIKTOK_ACCOUNT_NAME=default
"""
    
    with open(config_dir / ".env.example", "w") as f:
        f.write(env_sample)
    
    # Sample README for setup
    readme_content = """# Multi-Platform Video Uploader Configuration

## Setup Instructions:

### 1. YouTube API Setup:
1. Go to Google Cloud Console: https://console.cloud.google.com/
2. Create a new project or select existing one
3. Enable YouTube Data API v3
4. Create OAuth 2.0 credentials (Desktop Application)
5. Download the JSON file and save as `config/youtube_credentials.json`

### 2. Twitter/X API Setup:
1. Go to https://developer.twitter.com/
2. Create a new app
3. Generate API keys and access tokens
4. Copy `.env.example` to `.env` and fill in your credentials

### 3. TikTok Setup:
1. Install: `pip install tiktokautouploader`
2. First run will require browser login to TikTok
3. Account credentials will be saved for future uploads

### 4. Environment Variables:
Copy `.env.example` to `.env` and update with your credentials.

### Usage Examples:
```bash
# Upload to all platforms
python multi_platform_uploader.py --video "video.mp4" --title "My Video" --description "Description" --platforms youtube twitter tiktok

# Upload to specific platforms only
python multi_platform_uploader.py --video "video.mp4" --title "My Video" --description "Description" --platforms youtube twitter

# Upload with additional options
python multi_platform_uploader.py --video "video.mp4" --title "My Video" --description "Description" --platforms youtube --youtube-category tech --youtube-privacy unlisted
```
"""
    
    with open(config_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Configuration files created in {config_dir}/")
    print("📝 Please read config/README.md for setup instructions")


def main():
    parser = argparse.ArgumentParser(description="Upload videos to multiple platforms")
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--title', help='Video title')
    parser.add_argument('--description', help='Video description')
    parser.add_argument('--platforms', nargs='+', choices=PLATFORMS, 
                       default=PLATFORMS, help='Platforms to upload to')
    
    # YouTube specific options
    parser.add_argument('--youtube-category', choices=list(YOUTUBE_CATEGORIES.keys()),
                       default='people', help='YouTube video category')
    parser.add_argument('--youtube-privacy', choices=['public', 'private', 'unlisted'],
                       default='public', help='YouTube privacy setting')
    parser.add_argument('--youtube-tags', nargs='*', help='YouTube tags')
    
    # TikTok specific options
    parser.add_argument('--tiktok-account', default='default', 
                       help='TikTok account name (for multiple accounts)')
    
    # Utility options
    parser.add_argument('--setup', action='store_true', 
                       help='Create sample configuration files')
    
    args = parser.parse_args()
    
    if args.setup:
        create_sample_config()
        return
    
    # Check required arguments if not in setup mode
    if not args.video or not args.title or not args.description:
        parser.error("--video, --title, and --description are required when not using --setup")
    
    # Load environment variables
    env_file = Path("config/.env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Initialize uploader
    uploader = MultiPlatformUploader()
    
    # Perform uploads
    results = uploader.upload_to_platforms(
        video_path=args.video,
        title=args.title,
        description=args.description,
        platforms=args.platforms,
        youtube_category=args.youtube_category,
        youtube_privacy=args.youtube_privacy,
        youtube_tags=args.youtube_tags,
        tiktok_account=args.tiktok_account
    )
    
    # Print results
    print("\n" + "="*50)
    print("UPLOAD RESULTS")
    print("="*50)
    
    for platform, result in results.items():
        status = "✅ SUCCESS" if result.get('success') else "❌ FAILED"
        print(f"\n{platform.upper()}: {status}")
        
        if result.get('success'):
            if 'url' in result:
                print(f"   URL: {result['url']}")
            if 'message' in result:
                print(f"   Info: {result['message']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    main() 