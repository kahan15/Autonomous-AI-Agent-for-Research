from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
import time
from urllib.parse import urlparse, quote_plus
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import feedparser
import hashlib

from .base import ResearchAgent, PlanStep, ExecutionResult

# Load environment variables
load_dotenv()

class SourceType:
    ACADEMIC = "academic"
    TECH_BLOG = "tech_blog"
    NEWS = "news"
    DOCUMENTATION = "documentation"

class Source:
    def __init__(self, url: str, title: str, snippet: str, source_type: str, 
                 credibility_score: float = 0.0, published_date: Optional[str] = None):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.source_type = source_type
        self.credibility_score = credibility_score
        self.published_date = published_date
        self.content = None
        self.hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate a unique hash for the source based on URL and title."""
        return hashlib.md5(f"{self.url}{self.title}".encode()).hexdigest()

class WebResearchAgent(ResearchAgent):
    """Specialized agent for conducting web-based research."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the web research agent."""
        super().__init__(config_path)
        self.session = requests.Session()
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Set up logging to file
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create a file handler
        log_file = log_dir / f"research_agent_{time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the file handler to the logger
        self.logger.addHandler(file_handler)
        self.logger.info(f"Logging to file: {log_file}")
        
        self.sources: List[Source] = []
        self.extracted_content = []
        self.analysis_results = {}
        self.current_task = None
        
        # API key
        self.search_api_key = os.getenv("SEARCH_API_KEY")
        if not self.search_api_key:
            self.logger.warning("SEARCH_API_KEY not found in environment variables")

        # Source type weights for diversity scoring
        self.source_weights = self.config['agent']['source_weights']

    def _gather_sources(self) -> ExecutionResult:
        """Gather sources using Serper API."""
        try:
            if not self.search_api_key:
                return ExecutionResult(
                    success=False,
                    error="SEARCH_API_KEY not found in environment variables"
                )

            gathered_sources = []
            categories = self.config['agent']['sources']['web']['categories']
            
            for category in categories:
                # Adjust search query based on category
                category_query = self._build_category_query(self.current_task, category)
                self.logger.info(f"Searching for category {category} with query: {category_query}")
                
                # Search using Serper API
                results = self._serper_search(category_query)
                
                if results:
                    for result in results:
                        if 'link' in result and 'title' in result:
                            source = Source(
                                url=result['link'],
                                title=result['title'],
                                snippet=result.get('snippet', ''),
                                source_type=category,
                                credibility_score=self._calculate_initial_credibility(result),
                                published_date=result.get('date')
                            )
                            gathered_sources.append(source)
                            self.logger.info(f"Added source: {source.url}")
                else:
                    self.logger.warning(f"No results found for category: {category}")
            
            if not gathered_sources:
                return ExecutionResult(
                    success=False,
                    error="No sources found for any category"
                )
            
            self.sources = gathered_sources
            self._save_sources()  # Save sources after gathering
            
            return ExecutionResult(
                success=True,
                output=f"Gathered {len(gathered_sources)} sources",
                metrics={
                    'sources_count': len(gathered_sources),
                    'categories_with_results': len([cat for cat in categories if any(s.source_type == cat for s in gathered_sources)])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error gathering sources: {str(e)}")
            return ExecutionResult(
                success=False,
                error=f"Failed to gather sources: {str(e)}"
            )

    def _build_category_query(self, task: str, category: str) -> str:
        """Build a search query based on the task and source category."""
        base_query = task.strip()
        
        if category == SourceType.ACADEMIC:
            return f"{base_query} site:arxiv.org OR site:research.google.com OR site:openai.com/research"
        elif category == SourceType.TECH_BLOG:
            return f"{base_query} site:ai.googleblog.com OR site:openai.com/blog OR site:huggingface.co/blog"
        elif category == SourceType.NEWS:
            return f"{base_query} site:medium.com OR site:towardsdatascience.com"
        elif category == SourceType.DOCUMENTATION:
            return f"{base_query} site:pytorch.org OR site:tensorflow.org OR site:huggingface.co/docs"
        
        return base_query

    def _serper_search(self, query: str) -> List[Dict]:
        """Perform web search using Serper API."""
        try:
            headers = {
                'X-API-KEY': self.search_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'q': query,
                'gl': 'us',  # Set region to US for consistent results
                'hl': 'en',  # Set language to English
                'num': self.config['agent']['sources']['web']['max_results']
            }
            
            response = requests.post(
                'https://google.serper.dev/search',
                headers=headers,
                json=payload,
                timeout=30  # Add timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"Search successful for query: {query}")
                
                # Extract organic results
                results = data.get('organic', [])
                if not results:
                    self.logger.warning(f"No organic results found for query: {query}")
                    
                # Log the number of results found
                self.logger.info(f"Found {len(results)} results for query: {query}")
                return results
            else:
                self.logger.error(f"Serper API error: Status {response.status_code}, Response: {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error in Serper search: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in Serper search: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in Serper search: {str(e)}")
            return []

    def _calculate_initial_credibility(self, result: Dict) -> float:
        """Calculate initial credibility score for a source."""
        score = 0.5  # Base score
        
        # Add score for HTTPS
        if result.get('link', '').startswith('https'):
            score += 0.1
            
        # Add score for known reliable domains
        domain = urlparse(result.get('link', '')).netloc
        if any(reliable in domain for reliable in ['.edu', '.gov', '.org', 'research.', 'blog.']):
            score += 0.2
            
        # Add score for recent content
        if result.get('date'):
            try:
                date_str = result['date']
                if '2024' in date_str or '2023' in date_str:
                    score += 0.2
            except:
                pass
            
        return min(score, 1.0)

    def plan(self, task: str) -> List[PlanStep]:
        """Create a research plan for the given task."""
        self.current_task = task
        self.logger.info(f"Planning phase started")
        
        # Define the research plan steps
        plan = [
            PlanStep(objective="Initialize research session"),
            PlanStep(objective="Gather diverse sources"),
            PlanStep(objective="Validate and score sources"),
            PlanStep(objective="Extract and validate content"),
            PlanStep(objective="Analyze and synthesize"),
            PlanStep(objective="Generate comprehensive report")
        ]
        
        return plan

    def execute_task(self, task: str) -> ExecutionResult:
        """Execute a research task."""
        try:
            self.current_task = task
            self.logger.info(f"Starting new task: {task}")
            
            # Planning phase
            self.logger.info("Planning phase started")
            plan = self.plan(task)
            
            # Execution phase
            self.logger.info("Execution phase started")
            for step in plan:
                self.logger.info(f"Executing step: {step}")
                result = self.execute(step)
                if not result.success:
                    return result
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            return ExecutionResult(
                success=False,
                output=f"Task execution failed: {str(e)}"
            )

    def execute(self, step: PlanStep) -> ExecutionResult:
        """Execute a research step."""
        try:
            self.logger.info(f"Executing step: {step}")
            
            if step.objective == "Initialize research session":
                return self._initialize_session()
                
            elif step.objective == "Gather diverse sources":
                result = self._gather_sources()
                return result  # Sources are now saved in _gather_sources
                
            elif step.objective == "Validate and score sources":
                if not self.sources:
                    self.logger.warning("No sources to validate")
                    return ExecutionResult(
                        success=False,
                        error="No sources available for validation"
                    )
                validated_sources = self._validate_sources(self.sources)
                self.sources = validated_sources
                self._save_sources()  # Save updated sources after validation
                return ExecutionResult(
                    success=True,
                    output=validated_sources,
                    metrics={'validated_count': len(validated_sources)}
                )
                
            elif step.objective == "Extract and validate content":
                if not self.sources:
                    self.logger.warning("No sources available, gathering sources first")
                    gather_result = self._gather_sources()
                    if not gather_result.success:
                        return gather_result
                
                extract_result = self._extract_content()
                if extract_result.success:
                    self._save_extracted_content()  # Save extracted content
                return extract_result
                
            elif step.objective == "Analyze and synthesize":
                if not self.extracted_content:
                    self.logger.warning("No content available, extracting content first")
                    extract_result = self._extract_content()
                    if not extract_result.success:
                        return extract_result
                
                analysis_result = self._analyze_content()
                if analysis_result.success:
                    self._save_analysis(analysis_result.output)  # Save analysis results
                return analysis_result
                
            elif step.objective == "Generate comprehensive report":
                return self._generate_report()
                
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unknown step objective: {step.objective}"
                )
                
        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e)
            )

    def reflect(self, execution_result: ExecutionResult) -> bool:
        """Reflect on the execution result and determine if replanning is needed."""
        if not execution_result.success:
            self.reflection_notes.append(f"Step failed: {execution_result.error}")
            return True
        
        if execution_result.metrics:
            confidence = execution_result.metrics.get('confidence', 1.0)
            sources_count = execution_result.metrics.get('sources_count', 0)
            content_size = execution_result.metrics.get('content_size', 0)
            
            if confidence < self.config['agent']['workflow']['reflection_threshold']:
                self.reflection_notes.append(f"Low confidence ({confidence:.2%}), suggesting replanning")
                return True
            
            if sources_count == 0:
                self.reflection_notes.append("No sources found, need to adjust search strategy")
                return True
            
            if content_size == 0:
                self.reflection_notes.append("No content extracted, need to review sources")
                return True
        
        self.reflection_notes.append("Step completed successfully")
        return False

    def _initialize_session(self) -> ExecutionResult:
        """Initialize the research session with proper configurations."""
        try:
            # Set up rate limiting and other session parameters
            self.session.headers.update({
                'User-Agent': 'Research Agent/1.0 (Academic Research)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            })
            
            # Clear any previous state
            self.sources = []
            self.extracted_content = []
            self.analysis_results = {}
            
            return ExecutionResult(
                success=True,
                output="Session initialized",
                metrics={'session_status': 'ready'}
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Session initialization failed: {str(e)}"
            )

    def _calculate_diversity_score(self, sources: List[Source]) -> float:
        """Calculate diversity score based on source types and weights."""
        if not sources:
            return 0.0
            
        type_counts = {stype: 0 for stype in self.source_weights.keys()}
        for source in sources:
            if source.source_type in type_counts:
                type_counts[source.source_type] += 1
                
        weighted_sum = sum(count * self.source_weights[stype] for stype, count in type_counts.items())
        max_possible = len(sources) * max(self.source_weights.values())
        
        return weighted_sum / max_possible if max_possible > 0 else 0.0

    def _validate_sources(self, sources: List[Source]) -> List[Source]:
        """Validate and score sources."""
        validated_sources = []
        
        for source in sources:
            try:
                # Check if source is from allowed domains
                domain = urlparse(source.url).netloc
                allowed_domains = self.config.get('security', {}).get('allowed_domains', [])
                
                if any(allowed_domain in domain for allowed_domain in allowed_domains):
                    self.logger.info(f"Found matching domain: {domain}")
                    
                    # Check relevance to AI/ML
                    if any(term in source.title.lower() or term in source.snippet.lower() 
                          for term in ['ai', 'ml', 'language model', 'llm', 'transformer']):
                        
                        # Check recency for non-academic sources
                        if 'arxiv.org' not in domain:
                            if '2024' in source.title or '2024' in source.snippet:
                                validated_sources.append(source)
                        else:
                            self.logger.info(f"Allowing research domain: {domain}")
                            validated_sources.append(source)
                            
            except Exception as e:
                self.logger.warning(f"Error validating source {source.url}: {str(e)}")
                continue
                
        return validated_sources

    def _get_source_type_distribution(self, sources: List[Source]) -> Dict[str, int]:
        """Get distribution of source types."""
        distribution = {stype: 0 for stype in self.source_weights.keys()}
        for source in sources:
            if source.source_type in distribution:
                distribution[source.source_type] += 1
        return distribution

    def _extract_content(self) -> ExecutionResult:
        """Extract content from gathered sources."""
        try:
            extracted_data = []
            successful_extractions = 0
            
            for source in self.sources:
                try:
                    # Configure session for more lenient SSL verification
                    self.session.verify = False
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    
                    # Add user agent to avoid blocks
                    headers = {
                        'User-Agent': self.config['agent']['security']['user_agent']
                    }
                    
                    # Fetch the webpage content
                    response = self.session.get(
                        source.url,
                        timeout=30,
                        headers=headers
                    )
                    response.raise_for_status()
                    
                    # Parse content
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'aside']):
                        element.decompose()
                    
                    # Extract main content based on source type
                    if source.source_type == 'academic':
                        content = self._extract_academic_content(soup)
                    elif source.source_type == 'tech_blog':
                        content = self._extract_blog_content(soup)
                    else:
                        content = self._extract_general_content(soup)
                    
                    # Store extracted content
                    extracted_data.append({
                        'url': source.url,
                        'title': source.title,
                        'source_type': source.source_type,
                        'content': content,
                        'extraction_timestamp': time.time()
                    })
                    
                    successful_extractions += 1
                    self.logger.info(f"Successfully extracted content from {source.url}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract content from {source.url}: {str(e)}")
                    continue
            
            if not extracted_data:
                return ExecutionResult(
                    success=False,
                    error="Failed to extract content from any sources"
                )
            
            self.extracted_content = extracted_data
            
            return ExecutionResult(
                success=True,
                output=extracted_data,
                metrics={
                    'successful_extractions': successful_extractions,
                    'total_sources': len(self.sources),
                    'success_rate': successful_extractions / len(self.sources)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Content extraction failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e)
            )

    def _extract_academic_content(self, soup: BeautifulSoup) -> str:
        """Extract content from academic sources."""
        # Try to find abstract first
        abstract = soup.find('blockquote', {'class': 'abstract'})
        if abstract:
            return abstract.get_text(separator=' ', strip=True)
            
        # Look for specific academic paper sections
        sections = []
        for heading in ['abstract', 'introduction', 'conclusion']:
            section = soup.find(lambda tag: tag.name in ['div', 'section'] and 
                              heading in tag.get_text().lower())
            if section:
                sections.append(section.get_text(separator=' ', strip=True))
                
        if sections:
            return ' '.join(sections)
            
        # Fallback to main content
        return self._extract_general_content(soup)

    def _extract_blog_content(self, soup: BeautifulSoup) -> str:
        """Extract content from blog posts."""
        # Try common blog post containers
        content = soup.find('article') or \
                 soup.find('main') or \
                 soup.find('div', {'class': ['post-content', 'article-content', 'blog-post']})
                 
        if content:
            return content.get_text(separator=' ', strip=True)
            
        return self._extract_general_content(soup)

    def _extract_general_content(self, soup: BeautifulSoup) -> str:
        """Extract content from general web pages."""
        # Remove navigation, headers, footers, etc.
        for tag in soup(['nav', 'header', 'footer', 'aside']):
            tag.decompose()
            
        # Get the main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': 'content'})
        if main_content:
            return main_content.get_text(separator=' ', strip=True)
            
        # Fallback to body text
        return ' '.join(p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 100)

    def _analyze_content(self) -> ExecutionResult:
        """Analyze and synthesize extracted content."""
        try:
            analysis = {
                'key_findings': [],
                'trends': [],
                'metadata': {
                    'analysis_timestamp': time.time(),
                    'source_count': len(self.extracted_content)
                }
            }
            
            # Extract key findings and trends
            for content in self.extracted_content:
                text = content.get('content', '')
                
                # Look for key findings
                findings = []
                
                # Look for mentions of specific AI developments
                ai_terms = ['language model', 'llm', 'transformer', 'gpt', 'bert', 'ai model']
                for term in ai_terms:
                    if term in text.lower():
                        # Find the sentence containing the term
                        sentences = text.split('.')
                        for sentence in sentences:
                            if term in sentence.lower():
                                findings.append(sentence.strip())
                
                # Add source-specific findings
                if findings:
                    analysis['key_findings'].extend([{
                        'finding': finding,
                        'source': content.get('url'),
                        'confidence': 0.8
                    } for finding in findings[:3]])  # Limit to top 3 findings per source
                    
            # Identify trends across sources
            trend_keywords = {
                'performance': ['faster', 'efficient', 'improved', 'better'],
                'capabilities': ['new ability', 'can now', 'feature'],
                'applications': ['use case', 'application', 'industry'],
                'challenges': ['limitation', 'challenge', 'problem', 'issue']
            }
            
            for category, keywords in trend_keywords.items():
                category_findings = []
                for content in self.extracted_content:
                    text = content.get('content', '').lower()
                    for keyword in keywords:
                        if keyword in text:
                            sentences = text.split('.')
                            for sentence in sentences:
                                if keyword in sentence.lower():
                                    category_findings.append(sentence.strip())
                
                if category_findings:
                    analysis['trends'].append({
                        'category': category,
                        'findings': category_findings[:2],  # Limit to top 2 findings per category
                        'confidence': 0.7
                    })
            
            self.analysis_results = analysis
            
            return ExecutionResult(
                success=True,
                output=analysis,
                metrics={
                    'finding_count': len(analysis['key_findings']),
                    'trend_count': len(analysis['trends']),
                    'confidence': 0.8
                }
            )
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                output=None,
                error=str(e)
            )

    def _simulate_content(self) -> ExecutionResult:
        """Generate simulated content when extraction fails."""
        simulated_content = {
            'extracted_text': [
                {
                    'url': 'https://example.com/ai-trends-2024',
                    'title': 'Latest Developments in AI Language Models 2024',
                    'content': """
                    Large Language Models have become more efficient and accurate in 2024, with significant improvements in performance and capabilities.
                    Key developments include better context understanding, reduced hallucination rates, and more efficient training methods.
                    New applications in healthcare, scientific research, and education show promising results.
                    Ethical considerations and responsible AI development remain important focus areas.
                    """,
                    'timestamp': time.time()
                }
            ],
            'metadata': {
                'extraction_timestamp': time.time(),
                'success_rate': 1.0,
                'is_simulated': True
            }
        }
        
        self.extracted_content = simulated_content['extracted_text']
        
        return ExecutionResult(
            success=True,
            output=simulated_content,
            metrics={
                'extracted_count': 1,
                'success_rate': 1.0,
                'is_simulated': True
            }
        )

    def _generate_report(self) -> ExecutionResult:
        """Generate a comprehensive research report."""
        try:
            report = {
                'title': 'Research Report',
                'query': self.current_task,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'findings': [],
                'summary': 'Analysis of current trends and developments in AI',
                'sources': []
            }
            
            # Add key findings
            if self.analysis_results and 'key_findings' in self.analysis_results:
                for finding in self.analysis_results['key_findings']:
                    report['findings'].append({
                        'title': finding.get('finding', '')[:100] + '...',
                        'content': finding.get('finding', ''),
                        'source': finding.get('source', ''),
                        'confidence': finding.get('confidence', 0.0)
                    })
            
            # Add sources
            if self.sources:
                report['sources'] = [source.url for source in self.sources]
            
            # Save report
            report_dir = Path('reports')
            report_dir.mkdir(exist_ok=True)
            
            # Save as JSON
            json_path = report_dir / 'research_report.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Save as Markdown
            md_path = report_dir / 'research_report.md'
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {report['title']}\n\n")
                f.write(f"## Query\n{report['query']}\n\n")
                f.write(f"## Timestamp\n{report['timestamp']}\n\n")
                
                f.write("## Key Findings\n")
                for finding in report['findings']:
                    f.write(f"### {finding['title']}\n")
                    f.write(f"{finding['content']}\n")
                    f.write(f"Source: {finding['source']}\n")
                    f.write(f"Confidence: {finding['confidence']:.2f}\n\n")
                
                f.write("## Sources\n")
                for source in report['sources']:
                    f.write(f"- {source}\n")
            
            return ExecutionResult(
                success=True,
                output=report,
                metrics={
                    'finding_count': len(report['findings']),
                    'source_count': len(report['sources']),
                    'confidence': 0.9
                }
            )
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=str(e)
            )

    def _find_supporting_evidence(self, finding: str, text: str) -> str:
        """Find text evidence that supports a finding."""
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        finding_lower = finding.lower()
        
        # Split finding into key terms
        key_terms = finding_lower.split()
        
        # Find paragraphs containing most key terms
        paragraphs = text.split('\n')
        best_paragraph = ""
        max_term_count = 0
        
        for para in paragraphs:
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue
                
            para_lower = para.lower()
            term_count = sum(1 for term in key_terms if term in para_lower)
            
            if term_count > max_term_count:
                max_term_count = term_count
                best_paragraph = para.strip()
        
        # Return evidence if we found a good match
        if max_term_count >= len(key_terms) * 0.5:  # At least 50% of terms found
            return best_paragraph
        
        return ""

    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate a markdown version of the report."""
        md = []
        
        # Title and summary
        md.append(f"# {report['title']}")
        md.append(f"\nGenerated on: {datetime.fromtimestamp(report['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"\n## Research Task")
        md.append(f"\n{report['task']}")
        md.append(f"\n## Summary")
        md.append(f"\n{report['summary']}")
        
        # Sections with findings and citations
        for section in report["sections"]:
            md.append(f"\n## {section['title']}")
            
            for i, finding in enumerate(section["content"]):
                md.append(f"\n### Finding {i+1}")
                md.append(f"\n{finding}")
                
                # Add citations for this finding
                citations = [c for c in section["citations"] if c["finding_index"] == i]
                if citations:
                    md.append("\n#### Supporting Evidence:")
                    for citation in citations:
                        md.append(f"\n> {citation['text']}")
                        md.append(f"\n> Source: [{citation['source_url']}]({citation['source_url']})")
        
        # Sources section
        md.append("\n## Sources")
        for source in report["sources"]:
            md.append(f"\n- [{source['title']}]({source['url']}) (Accessed: {source['accessed_date']})")
        
        return "\n".join(md)

    def get_latest_log(self) -> Optional[str]:
        """Get the contents of the latest log file."""
        try:
            log_dir = self.data_dir / "logs"
            if not log_dir.exists():
                return None
                
            # Get all log files
            log_files = list(log_dir.glob("research_agent_*.log"))
            if not log_files:
                return None
                
            # Get the most recent log file
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            
            # Read and return its contents
            with open(latest_log, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            self.logger.error(f"Error reading log file: {str(e)}")
            return None

    def _is_allowed_domain(self, url: str) -> bool:
        """Check if the domain is in the allowed list."""
        try:
            if not url:
                return False
                
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            # Log the domain being checked
            self.logger.debug(f"Checking domain: {domain}")
            
            # Special cases for academic and research institutions
            if any(edu_suffix in domain for edu_suffix in ['.edu', '.ac.', 'university', 'institute']):
                self.logger.info(f"Allowing academic domain: {domain}")
                return True
                
            # Special cases for research and preprint servers
            if any(research_domain in domain for research_domain in ['arxiv', 'biorxiv', 'medrxiv', 'researchgate', 'sciencedirect']):
                self.logger.info(f"Allowing research domain: {domain}")
                return True
                
            # Special cases for AI company blogs
            if any(company_blog in domain for company_blog in ['blog.openai', 'ai.googleblog', 'research.google', 'ai.facebook', 'deepmind.com/blog']):
                self.logger.info(f"Allowing company blog domain: {domain}")
                return True
                
            # Special cases for AI community sites
            if any(community_site in domain for community_site in ['reddit.com/r/MachineLearning', 'reddit.com/r/artificial', 'news.ycombinator']):
                self.logger.info(f"Allowing community domain: {domain}")
                return True
                
            # Check if domain or any parent domain is in allowed list
            domain_parts = domain.split('.')
            for i in range(len(domain_parts) - 1):
                check_domain = '.'.join(domain_parts[i:])
                if check_domain in self.config['security']['allowed_domains']:
                    self.logger.info(f"Found matching domain: {check_domain}")
                    return True
                    
            # Check for exact match
            if domain in self.config['security']['allowed_domains']:
                self.logger.info(f"Found exact domain match: {domain}")
                return True
                
            self.logger.debug(f"Domain not allowed: {domain}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking domain: {str(e)}")
            return False

    def _save_to_json(self, data: Dict, filename: str) -> None:
        """Save data to a JSON file in the data directory."""
        try:
            file_path = self.data_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving data to {filename}: {str(e)}")

    def _save_sources(self) -> None:
        """Save gathered sources to sources.json."""
        try:
            sources_data = []
            for source in self.sources:
                source_dict = {
                    'url': source.url,
                    'title': source.title,
                    'snippet': source.snippet,
                    'source_type': source.source_type,
                    'credibility_score': source.credibility_score,
                    'published_date': source.published_date,
                    'hash': source.hash
                }
                sources_data.append(source_dict)
            
            if sources_data:  # Only save if we have sources
                self._save_to_json(sources_data, 'sources.json')
                self.logger.info(f"Saved {len(sources_data)} sources to sources.json")
            else:
                self.logger.warning("No sources to save")
        except Exception as e:
            self.logger.error(f"Error saving sources: {str(e)}")

    def _save_extracted_content(self) -> None:
        """Save extracted content to extracted_content.json."""
        self._save_to_json(self.extracted_content, 'extracted_content.json')

    def _save_analysis(self, analysis: Dict) -> None:
        """Save analysis results to analysis.json."""
        self._save_to_json(analysis, 'analysis.json') 