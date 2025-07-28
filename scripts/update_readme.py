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
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            logger.warning("GEMINI_API_KEY not found. AI summaries will be disabled.")
            self.model = None

    def generate_ai_summary(self, content: str, context: str = "") -> str:
        """Generate AI summary using Gemini"""
        if not self.model:
            return content

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

            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return content

    def generate_learning_summary(self, content: str) -> Dict[str, str]:
        """Generate AI summary for learning content"""
        if not self.model:
            return {"title": "Learning Update", "summary": content[:100] + "..."}

        try:
            # Extract URLs from content
            urls = re.findall(r'https?://[^\s]+', content)

            # Create prompt for learning summary
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
            summary = summary_match.group(1).strip() if summary_match else content[:100] + "..."

            return {
                "title": title,
                "summary": summary,
                "urls": urls
            }
        except Exception as e:
            logger.error(f"Error generating learning summary: {e}")
            return {
                "title": "Learning Update",
                "summary": content[:100] + "...",
                "urls": re.findall(r'https?://[^\s]+', content)
            }

    def parse_learning_tracker(self) -> List[Dict[str, Any]]:
        """Parse the learning tracker file with new format"""
        try:
            if not os.path.exists('learning_tracker.txt'):
                return []

            with open('learning_tracker.txt', 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all learning entries with date format <DD-MM-YY>
            pattern = r'<(\d{2}-\d{2}-\d{2})>\s*(.*?)\s*</\d{2}-\d{2}-\d{2}>'
            matches = re.findall(pattern, content, re.DOTALL)

            learnings = []
            for date_str, learning_content in matches[:self.max_learning_entries]:  # Get latest N entries
                if learning_content.strip():
                    # Generate AI summary for the learning
                    learning_data = self.generate_learning_summary(learning_content.strip())

                    learnings.append({
                        'date': date_str,
                        'title': learning_data['title'],
                        'summary': learning_data['summary'],
                        'urls': learning_data['urls'],
                        'raw_content': learning_content.strip()
                    })

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
        markdown = f"# GitHub Activity for {self.username}\n\n"

        # Exploring Section (at the top) - only show if has content
        if learnings:
            markdown += "## Exploring...\n"
            for learning in learnings:
                markdown += f"- **{learning['title']}** ({learning['date']})\n"
                markdown += f"  {learning['summary']}\n"

                # Add hyperlinks if any URLs found
                if learning['urls']:
                    for url in learning['urls']:
                        # Try to extract a meaningful name from URL or use generic text
                        link_text = "Resource"
                        if 'github.com' in url:
                            link_text = "GitHub"
                        elif 'youtube.com' in url or 'youtu.be' in url:
                            link_text = "Video"
                        elif 'docs.' in url or 'documentation' in url:
                            link_text = "Documentation"
                        elif 'tutorial' in url:
                            link_text = "Tutorial"

                        markdown += f"  [{link_text}]({url})\n"
                markdown += "\n"



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
