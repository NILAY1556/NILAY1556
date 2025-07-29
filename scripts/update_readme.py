#!/usr/bin/env python3
"""
GitHub Profile README Auto-Updater with AI Summaries
Fetches GitHub activity and generates AI-powered summaries using Google Gemini
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from github import Github
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubActivityFetcher:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.username = os.getenv('GITHUB_USERNAME', 'NILAY1556')

        # Configuration for "this month" timeframes
        self.github_activity_days = 30  # GitHub activity from this month (last 30 days)
        self.max_learning_entries = 3  # Show latest 3 learning entries
        self.max_events_to_process = 100  # Process last 100 GitHub events
        self.max_items_per_section = 5  # Max items to show per activity section

        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")

        self.github = Github(self.github_token)

        # Configure Gemini AI
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
            logger.info("Gemini AI model configured successfully.")
        else:
            logger.warning("GEMINI_API_KEY not found. AI summaries will be disabled.")
            self.model = None

    def generate_ai_summary(self, content: str, context: str = "") -> str:
        """Generate AI summary using Gemini"""
        if not self.model:
            logger.info("Using fallback summary (no AI model available)")
            return self._create_fallback_summary(content)

        try:
            prompt = f"""You are a GitHub activity summarizer. For the activity from @{self.username}:

            Guidelines for summarization:
            - Summarize the content in exactly 75 words
            - Provide the summary from the perspective of "@{self.username} has"
            - Focus on technical details and key contributions
            - Do not use any emojis or icons
            - Be professional and precise
            - If content contains code, extract the key message while ignoring specific code details
            - Do not refer to external links in the summary

            Context: {context}
            Content to summarize: {content}"""

            logger.info(f"Generating AI summary for content length: {len(content)} chars")
            response = self.model.generate_content(prompt)
            ai_summary = response.text.strip()
            logger.info(f"AI summary generated: {ai_summary[:100]}...")
            return ai_summary
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return self._create_fallback_summary(content)

    def _create_fallback_summary(self, content: str) -> str:
        """Create a basic summary when AI is not available"""
        if not content:
            return f"@{self.username} has made a contribution."

        # Clean up the content - remove markdown, excessive whitespace, and common PR template text
        clean_content = re.sub(r'#+\s*', '', content)  # Remove markdown headers
        clean_content = re.sub(r'\[.\]\s*', '', clean_content)  # Remove checkboxes
        clean_content = re.sub(r'##\s*(Description|Type of change|How Has This Been Tested|Checklist).*?(?=##|$)', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
        clean_content = re.sub(r'Fixes\s*#\d+', '', clean_content, flags=re.IGNORECASE)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()

        # Extract the first meaningful sentence or take first 100 characters
        sentences = [s.strip() for s in clean_content.split('.') if s.strip() and len(s.strip()) > 10]
        if sentences:
            summary = sentences[0]
            if len(summary) > 100:
                summary = summary[:97] + "..."
        else:
            summary = clean_content[:97] + "..." if len(clean_content) > 100 else clean_content

        return f"@{self.username} has {summary.lower()}" if summary else f"@{self.username} has made a contribution."

    def generate_learning_summary(self, content: str) -> Dict[str, str]:
        """Generate AI summary for learning content"""
        # Process inline hyperlinks first (format: text[url])
        processed_content = self._process_inline_hyperlinks(content)

        if not self.model:
            return {
                "title": "Learning Update",
                "summary": processed_content[:100] + "...",
                "processed_content": processed_content
            }

        try:
            # Create prompt for learning summary using processed content
            prompt = f"""Analyze this learning content and provide:
            1. A concise title (3-5 words) that captures the main topic
            2. A clean, professional 1-2 line summary (max 150 characters)

            Guidelines:
            - Focus on the key technology, concept, or skill learned
            - Be specific about what was accomplished or understood
            - No emojis or icons
            - Professional tone
            - If multiple topics, focus on the most significant one

            Learning content: {content}

            Respond in this exact format:
            TITLE: [your title here]
            SUMMARY: [your summary here]"""

            response = self.model.generate_content(prompt)
            result = response.text.strip()

            # Parse the response
            title_match = re.search(r'TITLE:\s*(.+)', result)
            summary_match = re.search(r'SUMMARY:\s*(.+)', result)

            title = title_match.group(1).strip() if title_match else "Learning Update"
            summary = summary_match.group(1).strip() if summary_match else processed_content[:100] + "..."

            return {
                "title": title,
                "summary": summary,
                "processed_content": processed_content
            }
        except Exception as e:
            logger.error(f"Error generating learning summary: {e}")
            return {
                "title": "Learning Update",
                "summary": processed_content[:100] + "...",
                "processed_content": processed_content
            }

    def _process_inline_hyperlinks(self, content: str) -> str:
        """Convert inline hyperlinks from format 'text[url]' to markdown '[text](url)'"""
        # Pattern to match text[url] format
        # This matches: any_text_without_spaces[http://url] or any_text_without_spaces[https://url]
        # Updated pattern to be more flexible with text before the bracket
        pattern = r'([a-zA-Z0-9_\-\.]+)\[(https?://[^\]]*)\]'

        def replace_link(match):
            text = match.group(1)
            url = match.group(2)
            # Only create link if URL is not empty
            if url.strip():
                return f'[{text}]({url})'
            else:
                # If URL is empty, just return the text
                return text

        # Replace all matches with markdown format
        processed = re.sub(pattern, replace_link, content)

        # Also handle any remaining standalone URLs that weren't in the text[url] format
        # Convert standalone URLs to generic links, but avoid URLs that are already in markdown format
        standalone_url_pattern = r'(?<!\]\()(https?://[^\s\[\]]+)(?!\))'

        def replace_standalone_url(match):
            url = match.group(1)
            # Try to extract a meaningful name from URL
            if 'github.com' in url:
                return f'[GitHub]({url})'
            elif 'youtube.com' in url or 'youtu.be' in url:
                return f'[Video]({url})'
            elif 'docs.' in url or 'documentation' in url:
                return f'[Documentation]({url})'
            elif 'arxiv.org' in url:
                return f'[Paper]({url})'
            elif 'tutorial' in url:
                return f'[Tutorial]({url})'
            else:
                return f'[Resource]({url})'

        # Only process URLs that are not already part of markdown links
        processed = re.sub(standalone_url_pattern, replace_standalone_url, processed)

        return processed

    def parse_learning_tracker(self) -> List[Dict[str, Any]]:
        """Parse the learning tracker file with new format"""
        try:
            if not os.path.exists('learning_tracker.txt'):
                logger.warning("learning_tracker.txt not found")
                return []

            with open('learning_tracker.txt', 'r', encoding='utf-8') as f:
                content = f.read()

            logger.info(f"Learning tracker content length: {len(content)} chars")

            # Find all learning entries with date format <DD-MM-YY>
            pattern = r'<(\d{2}-\d{2}-\d{2})>\s*(.*?)\s*</\d{2}-\d{2}-\d{2}>'
            matches = re.findall(pattern, content, re.DOTALL)

            logger.info(f"Found {len(matches)} learning entries")

            learnings = []
            for date_str, learning_content in matches[:self.max_learning_entries]:  # Get latest N entries
                if learning_content.strip():
                    logger.info(f"Processing learning entry for {date_str}")
                    # Generate AI summary for the learning
                    learning_data = self.generate_learning_summary(learning_content.strip())

                    learnings.append({
                        'date': date_str,
                        'title': learning_data['title'],
                        'summary': learning_data['summary'],
                        'processed_content': learning_data['processed_content'],
                        'raw_content': learning_content.strip()
                    })

            logger.info(f"Parsed {len(learnings)} learning entries successfully")
            return learnings
        except Exception as e:
            logger.error(f"Error parsing learning tracker: {e}")
            return []

    def get_recent_activity(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent GitHub activity using events API"""
        try:
            user = self.github.get_user(self.username)
            events = user.get_events()

            activities = {
                "pull_requests": [],
                "starred_repos": [],
                "forked_repos": []
            }

            cutoff_date = datetime.now() - timedelta(days=self.github_activity_days)

            for event in events[:self.max_events_to_process]:  # Process last N events
                try:
                    event_date = event.created_at
                    if event_date < cutoff_date:
                        continue

                    if event.type == 'PullRequestEvent' and event.payload.get('action') == 'opened':
                        pr_body = event.payload['pull_request'].get('body', '')
                        if pr_body:
                            ai_summary = self.generate_ai_summary(
                                pr_body,
                                f"Pull Request in {event.repo.full_name}"
                            )
                            activities['pull_requests'].append({
                                'repo': event.repo.full_name,
                                'pr_title': event.payload['pull_request']['title'],
                                'pr_url': event.payload['pull_request']['html_url'],
                                'ai_summary': ai_summary,
                                'date': event_date.strftime("%Y-%m-%d")
                            })

                    elif event.type == 'WatchEvent' and event.payload.get('action') == 'started':
                        activities['starred_repos'].append({
                            'repo': event.repo.full_name,
                            'repo_link': f"https://github.com/{event.repo.full_name}",
                            'date': event_date.strftime("%Y-%m-%d")
                        })

                    elif event.type == 'ForkEvent':
                        activities['forked_repos'].append({
                            'repo': event.repo.full_name,
                            'fork_url': event.payload['forkee']['html_url'],
                            'date': event_date.strftime("%Y-%m-%d")
                        })

                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    continue

            # Limit each activity type to max items per section
            for key in activities:
                activities[key] = activities[key][:self.max_items_per_section]

            return activities
        except Exception as e:
            logger.error(f"Error fetching GitHub activity: {e}")
            return {
                "recent_comments": [],
                "issues_raised": [],
                "pull_requests": [],
                "starred_repos": [],
                "forked_repos": []
            }

    def generate_markdown(self, activities: Dict[str, List[Dict[str, Any]]], learnings: List[Dict[str, Any]]) -> str:
        """Generate markdown content for README"""
        markdown = f"# {self.username}\n\n"

        logger.info(f"Generating markdown with {len(learnings)} learning entries")

        # Exploring Section (at the top) - only show if has content
        if learnings:
            logger.info("Adding Exploring section to markdown")
            markdown += "## Exploring...\n"
            for learning in learnings:
                logger.info(f"Adding learning: {learning['title']}")
                markdown += f"- **{learning['title']}** ({learning['date']})\n"
                markdown += f"  {learning['summary']}\n"

                # Add hyperlinks from processed content if they exist
                if learning['processed_content'] and '[' in learning['processed_content'] and '](' in learning['processed_content']:
                    # Extract just the hyperlinks from processed content for display
                    hyperlinks = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', learning['processed_content'])
                    if hyperlinks:
                        for link_text, url in hyperlinks:
                            markdown += f"  [{link_text}]({url})\n"

                markdown += "\n"
        else:
            logger.info("No learning entries found, skipping Exploring section")



        # Pull Requests Section - only show if has content
        if activities['pull_requests']:
            markdown += "## Pull Requests\n"
            for pr in activities['pull_requests']:
                markdown += f"- Opened a [PR]({pr['pr_url']}) in [{pr['repo']}]: {pr['pr_title']} ({pr['date']})\n"
                markdown += f"  > AI Summary: {pr['ai_summary']}\n\n"

        # Starred Repositories Section - only show if has content
        if activities['starred_repos']:
            markdown += "## Starred Repositories\n"
            for star in activities['starred_repos']:
                markdown += f"- Starred [{star['repo']}]({star['repo_link']}) on {star['date']}\n"
            markdown += "\n"

        # Forked Repositories Section - only show if has content
        if activities['forked_repos']:
            markdown += "## Forked Repositories\n"
            for fork in activities['forked_repos']:
                markdown += f"- Forked [{fork['repo']}]({fork['fork_url']}) on {fork['date']}\n"
            markdown += "\n"

        return markdown

    def update_readme(self):
        """Main function to update README"""
        try:
            logger.info("Starting README update with AI summaries...")

            # Parse learning tracker
            logger.info("Parsing learning tracker...")
            learnings = self.parse_learning_tracker()

            # Fetch GitHub activity
            logger.info("Fetching GitHub activity...")
            activities = self.get_recent_activity()

            logger.info(f"Data fetched: {len(learnings)} explorations, "
                       f"{len(activities['pull_requests'])} PRs, "
                       f"{len(activities['starred_repos'])} stars, "
                       f"{len(activities['forked_repos'])} forks")

            # Generate markdown content
            markdown_content = self.generate_markdown(activities, learnings)
            logger.info("Generated markdown content successfully!")

            # Write to README.md
            with open('README.md', 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info("README updated successfully!")

        except Exception as e:
            logger.error(f"Error updating README: {e}")
            raise


def main():
    """Main entry point"""
    try:
        fetcher = GitHubActivityFetcher()
        fetcher.update_readme()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main()

