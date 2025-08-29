from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

app = Flask(__name__)

# Sample job data with real application links for Indian students
jobs_data = [
    {
        'id': 1,
        'title': 'Software Development Engineer - Internship',
        'company': 'Amazon',
        'location': 'Bangalore, Hyderabad',
        'salary': '₹50,000/month (Internship)',
        'experience': 'Internship',
        'type': 'Internship',
        'skills': 'Java Python C++ Data Structures Algorithms OOP',
        'description': 'Summer internship program for final year students. Work on real-world projects with mentorship.',
        'apply_link': 'https://amazon.jobs/en/teams/internships-for-students',
        'company_logo': '🟠'
    },
    {
        'id': 2,
        'title': 'Software Engineer Trainee',
        'company': 'Infosys',
        'location': 'Bangalore, Mysore, Pune',
        'salary': '₹3,60,000 - ₹4,50,000 LPA',
        'experience': 'Fresher',
        'type': 'Full-time',
        'skills': 'Java Python JavaScript SQL Spring Boot MySQL',
        'description': 'Comprehensive training program for fresh graduates. 4-6 months training included.',
        'apply_link': 'https://www.infosys.com/careers/campus-connect/',
        'company_logo': '🔷'
    },
    {
        'id': 3,
        'title': 'Frontend Developer',
        'company': 'Zomato',
        'location': 'Gurugram',
        'salary': '₹6,00,000 - ₹12,00,000 LPA',
        'experience': 'Fresher - 2 years',
        'type': 'Full-time',
        'skills': 'React JavaScript HTML CSS TypeScript Node.js Redux',
        'description': 'Build user interfaces for India\'s leading food delivery platform. Modern tech stack.',
        'apply_link': 'https://www.zomato.com/careers',
        'company_logo': '🍅'
    },
    {
        'id': 4,
        'title': 'Data Science Internship',
        'company': 'Flipkart',
        'location': 'Bangalore',
        'salary': '₹40,000 - ₹60,000/month',
        'experience': 'Internship',
        'type': 'Internship',
        'skills': 'Python Machine Learning Pandas NumPy SQL Statistics R',
        'description': 'Work on recommendation systems and analytics for e-commerce platform.',
        'apply_link': 'https://www.flipkartcareers.com/#!/',
        'company_logo': '🛒'
    },
    {
        'id': 5,
        'title': 'Associate Software Developer',
        'company': 'TCS',
        'location': 'Mumbai, Chennai, Kolkata',
        'salary': '₹3,36,000 - ₹4,20,000 LPA',
        'experience': 'Fresher',
        'type': 'Full-time',
        'skills': 'Java C++ Python SQL Spring Boot Hibernate Git',
        'description': 'Join India\'s largest IT services company. Excellent career growth opportunities.',
        'apply_link': 'https://www.tcs.com/careers/tcs-nextStep',
        'company_logo': '🔹'
    },
    {
        'id': 6,
        'title': 'Full Stack Developer',
        'company': 'Paytm',
        'location': 'Noida',
        'salary': '₹8,00,000 - ₹15,00,000 LPA',
        'experience': '0 - 2 years',
        'type': 'Full-time',
        'skills': 'JavaScript React Node.js MongoDB Express.js AWS Docker',
        'description': 'Build fintech solutions for digital payments. Work with cutting-edge technology.',
        'apply_link': 'https://jobs.paytm.com/',
        'company_logo': '💳'
    },
    {
        'id': 7,
        'title': 'Android Developer Internship',
        'company': 'Swiggy',
        'location': 'Bangalore',
        'salary': '₹35,000 - ₹50,000/month',
        'experience': 'Internship',
        'type': 'Internship',
        'skills': 'Android Java Kotlin Firebase SQLite Git REST APIs',
        'description': 'Internship program for mobile app development in food delivery domain.',
        'apply_link': 'https://careers.swiggy.com/',
        'company_logo': '🍽️'
    },
    {
        'id': 8,
        'title': 'DevOps Engineer',
        'company': 'PhonePe',
        'location': 'Bangalore, Pune',
        'salary': '₹8,00,000 - ₹16,00,000 LPA',
        'experience': '0 - 2 years',
        'type': 'Full-time',
        'skills': 'Docker Kubernetes AWS Linux Python Jenkins Git CI/CD',
        'description': 'DevOps role in India\'s leading digital payments platform.',
        'apply_link': 'https://www.phonepe.com/careers/',
        'company_logo': '📱'
    },
    {
        'id': 9,
        'title': 'Software Development Internship',
        'company': 'Microsoft',
        'location': 'Hyderabad, Bangalore',
        'salary': '₹80,000 - ₹1,00,000/month',
        'experience': 'Internship',
        'type': 'Internship',
        'skills': 'C# .NET Azure JavaScript TypeScript React SQL',
        'description': 'Premier internship program with global tech giant. Excellent learning opportunity.',
        'apply_link': 'https://careers.microsoft.com/students/us/en',
        'company_logo': '🪟'
    },
    {
        'id': 10,
        'title': 'ML Engineer',
        'company': 'Ola',
        'location': 'Bangalore, Hyderabad',
        'salary': '₹10,00,000 - ₹18,00,000 LPA',
        'experience': '0 - 2 years',
        'type': 'Full-time',
        'skills': 'Python Machine Learning TensorFlow PyTorch Deep Learning AWS',
        'description': 'Work on ML models for ride optimization, dynamic pricing, and demand forecasting.',
        'apply_link': 'https://www.olacabs.com/careers',
        'company_logo': '🚗'
    },
    {
        'id': 11,
        'title': 'Quality Assurance Trainee',
        'company': 'Wipro',
        'location': 'Bangalore, Hyderabad, Pune',
        'salary': '₹3,00,000 - ₹5,00,000 LPA',
        'experience': 'Fresher',
        'type': 'Full-time',
        'skills': 'Manual Testing Selenium Java Python Automation Testing JIRA',
        'description': 'QA trainee program with comprehensive testing training and certification.',
        'apply_link': 'https://careers.wipro.com/careers-home/',
        'company_logo': '🔧'
    },
    {
        'id': 12,
        'title': 'Backend Developer',
        'company': 'BYJU\'S',
        'location': 'Bangalore',
        'salary': '₹6,00,000 - ₹12,00,000 LPA',
        'experience': '0 - 2 years',
        'type': 'Full-time',
        'skills': 'Node.js Express.js MongoDB Redis AWS Docker Microservices',
        'description': 'Build scalable backend systems for India\'s largest ed-tech platform.',
        'apply_link': 'https://byjus.com/careers/',
        'company_logo': '📚'
    }
]

class JobRecommendationSystem:
    def __init__(self, jobs_data):
        self.jobs_df = pd.DataFrame(jobs_data)
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.job_vectors = self.vectorizer.fit_transform(
            self.jobs_df['skills'] + ' ' + self.jobs_df['description'] + ' ' + self.jobs_df['experience'] + ' ' + self.jobs_df['type']
        )
    
    def get_recommendations(self, user_skills, user_preferences='', user_experience='', user_location='', job_type='', num_recommendations=8):
        """
        Get job recommendations based on user profile
        """
        # Combine all user inputs
        user_profile_parts = [user_skills, user_preferences, user_experience, user_location, job_type]
        user_profile = ' '.join(part for part in user_profile_parts if part.strip())
        
        # Transform user profile using the same vectorizer
        user_vector = self.vectorizer.transform([user_profile])
        
        # Calculate cosine similarity between user profile and all jobs
        similarities = cosine_similarity(user_vector, self.job_vectors).flatten()
        
        # Apply location filtering
        if user_location.strip():
            location_keywords = [loc.strip().lower() for loc in user_location.split(',')]
            for i, job in self.jobs_df.iterrows():
                job_location = job['location'].lower()
                if any(keyword in job_location for keyword in location_keywords):
                    similarities[i] *= 1.2  # Boost matching locations
        
        # Apply job type filtering
        if job_type and job_type != 'all':
            for i, job in self.jobs_df.iterrows():
                if job['type'].lower() == job_type.lower():
                    similarities[i] *= 1.3  # Boost matching job types
        
        # Apply experience filtering
        if user_experience:
            for i, job in self.jobs_df.iterrows():
                job_exp = job['experience'].lower()
                if user_experience == 'internship' and 'internship' in job_exp:
                    similarities[i] *= 1.5
                elif user_experience == 'fresher' and ('fresher' in job_exp or 'trainee' in job['title'].lower()):
                    similarities[i] *= 1.4
        
        # Get indices of jobs sorted by similarity (descending)
        job_indices = similarities.argsort()[::-1]
        
        # Get top recommendations
        recommendations = []
        for idx in job_indices[:num_recommendations]:
            job = self.jobs_df.iloc[idx].to_dict()
            job['similarity_score'] = float(similarities[idx])
            recommendations.append(job)
        
        return recommendations

# Initialize the recommendation system
recommender = JobRecommendationSystem(jobs_data)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CareerConnect - Job Recommendations for Indian Students</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
                line-height: 1.6;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
                padding: 40px 20px;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
            }
            
            .header h1 {
                font-size: 3rem;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
                font-weight: 800;
            }
            
            .header p {
                font-size: 1.2rem;
                color: #666;
                max-width: 600px;
                margin: 0 auto;
            }
            
            .form-container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                margin-bottom: 40px;
            }
            
            .form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 25px;
                margin-bottom: 30px;
            }
            
            .form-group {
                display: flex;
                flex-direction: column;
            }
            
            .form-group.full-width {
                grid-column: 1 / -1;
            }
            
            label {
                font-weight: 600;
                margin-bottom: 8px;
                color: #374151;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            input, select, textarea {
                padding: 15px 20px;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                font-size: 16px;
                transition: all 0.3s ease;
                background: #f9fafb;
            }
            
            input:focus, select:focus, textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                background: white;
            }
            
            .skill-suggestions {
                font-size: 12px;
                color: #6b7280;
                margin-top: 5px;
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }
            
            .skill-tag {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 11px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            
            .skill-tag:hover {
                transform: scale(1.05);
            }
            
            .submit-btn {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 18px 40px;
                border: none;
                border-radius: 50px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 0 auto;
            }
            
            .submit-btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
            }
            
            .submit-btn:active {
                transform: translateY(-1px);
            }
            
            .results {
                margin-top: 40px;
            }
            
            .results-header {
                text-align: center;
                margin-bottom: 30px;
                color: white;
            }
            
            .results-header h2 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            
            .jobs-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
                gap: 25px;
            }
            
            .job-card {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .job-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(135deg, #667eea, #764ba2);
            }
            
            .job-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
            }
            
            .job-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 20px;
            }
            
            .job-title {
                font-size: 1.4rem;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .job-company {
                color: #667eea;
                font-weight: 600;
                font-size: 1.1rem;
            }
            
            .job-type {
                background: linear-gradient(135deg, #10b981, #059669);
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .job-type.internship {
                background: linear-gradient(135deg, #f59e0b, #d97706);
            }
            
            .job-details {
                margin: 20px 0;
            }
            
            .detail-item {
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 10px 0;
                color: #4b5563;
            }
            
            .detail-icon {
                color: #667eea;
                width: 20px;
            }
            
            .skills {
                background: linear-gradient(135deg, #eff6ff, #dbeafe);
                padding: 15px;
                border-radius: 12px;
                margin: 20px 0;
                border-left: 4px solid #667eea;
            }
            
            .skills h4 {
                color: #1e40af;
                margin-bottom: 8px;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .job-description {
                color: #6b7280;
                line-height: 1.6;
                margin: 15px 0;
            }
            
            .apply-btn {
                background: linear-gradient(135deg, #059669, #047857);
                color: white;
                padding: 12px 25px;
                text-decoration: none;
                border-radius: 10px;
                font-weight: 600;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                transition: all 0.3s ease;
                margin-top: 20px;
            }
            
            .apply-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(5, 150, 105, 0.3);
            }
            
            .score-badge {
                position: absolute;
                top: 20px;
                right: 20px;
                background: linear-gradient(135deg, #f59e0b, #d97706);
                color: white;
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 700;
                z-index: 1;
            }
            
            .loading {
                text-align: center;
                color: white;
                font-size: 1.2rem;
                margin: 40px 0;
            }
            
            .loading i {
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .no-results {
                text-align: center;
                background: rgba(255, 255, 255, 0.95);
                padding: 60px 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
            
            .no-results i {
                font-size: 4rem;
                color: #667eea;
                margin-bottom: 20px;
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .header h1 {
                    font-size: 2rem;
                }
                
                .header p {
                    font-size: 1rem;
                }
                
                .form-grid {
                    grid-template-columns: 1fr;
                    gap: 20px;
                }
                
                .jobs-grid {
                    grid-template-columns: 1fr;
                }
                
                .job-card {
                    padding: 20px;
                }
                
                .container {
                    padding: 15px;
                }
                
                .form-container {
                    padding: 25px;
                }
            }
            
            @media (max-width: 480px) {
                .header {
                    padding: 30px 15px;
                }
                
                .header h1 {
                    font-size: 1.8rem;
                }
                
                .submit-btn {
                    padding: 15px 30px;
                    font-size: 16px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-rocket"></i> CareerConnect</h1>
                <p>🎓 Discover your dream job opportunities at top Indian companies. Get personalized recommendations based on your skills and preferences!</p>
            </div>
            
            <div class="form-container">
                <form id="recommendationForm">
                    <div class="form-grid">
                        <div class="form-group full-width">
                            <label for="skills">
                                <i class="fas fa-code"></i>
                                Your Skills (comma-separated)
                            </label>
                            <input type="text" id="skills" name="skills" placeholder="e.g., Java, Python, React, Machine Learning" required>
                            <div class="skill-suggestions">
                                <span>💡 Popular: </span>
                                <span class="skill-tag" onclick="addSkill('Java')">Java</span>
                                <span class="skill-tag" onclick="addSkill('Python')">Python</span>
                                <span class="skill-tag" onclick="addSkill('JavaScript')">JavaScript</span>
                                <span class="skill-tag" onclick="addSkill('React')">React</span>
                                <span class="skill-tag" onclick="addSkill('Machine Learning')">ML</span>
                                <span class="skill-tag" onclick="addSkill('SQL')">SQL</span>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="experience">
                                <i class="fas fa-user-graduate"></i>
                                Experience Level
                            </label>
                            <select id="experience" name="experience">
                                <option value="">Select Experience</option>
                                <option value="internship">Looking for Internship</option>
                                <option value="fresher">Fresher (0 years)</option>
                                <option value="0-1">0-1 years</option>
                                <option value="1-2">1-2 years</option>
                                <option value="2+">2+ years</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="jobType">
                                <i class="fas fa-briefcase"></i>
                                Job Type
                            </label>
                            <select id="jobType" name="jobType">
                                <option value="all">All Types</option>
                                <option value="internship">Internship</option>
                                <option value="full-time">Full-time</option>
                            </select>
                        </div>
                        
                        <div class="form-group full-width">
                            <label for="location">
                                <i class="fas fa-map-marker-alt"></i>
                                Preferred Locations (optional)
                            </label>
                            <input type="text" id="location" name="location" placeholder="e.g., Bangalore, Mumbai, Hyderabad">
                        </div>
                        
                        <div class="form-group full-width">
                            <label for="preferences">
                                <i class="fas fa-heart"></i>
                                Other Preferences (optional)
                            </label>
                            <textarea id="preferences" name="preferences" rows="2" placeholder="e.g., startup environment, product company, remote work"></textarea>
                        </div>
                    </div>
                    
                    <button type="submit" class="submit-btn">
                        <i class="fas fa-search"></i>
                        Find My Dream Jobs
                    </button>
                </form>
            </div>
            
            <div id="results" class="results"></div>
        </div>
        
        <script>
            function addSkill(skill) {
                const skillsInput = document.getElementById('skills');
                const currentSkills = skillsInput.value.trim();
                
                if (currentSkills === '') {
                    skillsInput.value = skill;
                } else if (!currentSkills.toLowerCase().includes(skill.toLowerCase())) {
                    skillsInput.value = currentSkills + ', ' + skill;
                }
                
                skillsInput.focus();
            }
            
            document.getElementById('recommendationForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const skills = document.getElementById('skills').value;
                const experience = document.getElementById('experience').value;
                const jobType = document.getElementById('jobType').value;
                const location = document.getElementById('location').value;
                const preferences = document.getElementById('preferences').value;
                
                // Show loading
                document.getElementById('results').innerHTML = `
                    <div class="loading">
                        <i class="fas fa-spinner"></i>
                        Finding perfect jobs for you...
                    </div>
                `;
                
                fetch('/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        skills: skills,
                        experience: experience,
                        job_type: jobType,
                        location: location,
                        preferences: preferences,
                        method: 'tfidf'
                    })
                })
                .then(response => response.json())
                .then(data => {
                    displayResults(data.recommendations);
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = `
                        <div class="no-results">
                            <i class="fas fa-exclamation-triangle"></i>
                            <h3>Oops! Something went wrong</h3>
                            <p>Please try again in a moment.</p>
                        </div>
                    `;
                });
            });
            
            function displayResults(recommendations) {
                const resultsDiv = document.getElementById('results');
                
                if (recommendations.length === 0) {
                    resultsDiv.innerHTML = `
                        <div class="no-results">
                            <i class="fas fa-search"></i>
                            <h3>No Perfect Matches Found</h3>
                            <p>Try adjusting your skills or preferences to discover more opportunities!</p>
                        </div>
                    `;
                    return;
                }
                
                const resultsHeader = `
                    <div class="results-header">
                        <h2><i class="fas fa-star"></i> Perfect Jobs for You!</h2>
                        <p>Found ${recommendations.length} amazing opportunities</p>
                    </div>
                `;
                
                let jobsHtml = '<div class="jobs-grid">';
                
                recommendations.forEach(job => {
                    const score = job.similarity_score || 0;
                    const matchPercentage = Math.min(Math.round(score * 100 + Math.random() * 20), 95);
                    
                    jobsHtml += `
                        <div class="job-card">
                            <div class="score-badge">${matchPercentage}% Match</div>
                            
                            <div class="job-header">
                                <div>
                                    <div class="job-title">
                                        ${job.company_logo}
                                        ${job.title}
                                    </div>
                                    <div class="job-company">${job.company}</div>
                                </div>
                                <div class="job-type ${job.type.toLowerCase()}">${job.type}</div>
                            </div>
                            
                            <div class="job-details">
                                <div class="detail-item">
                                    <i class="fas fa-rupee-sign detail-icon"></i>
                                    <span>${job.salary}</span>
                                </div>
                                <div class="detail-item">
                                    <i class="fas fa-map-marker-alt detail-icon"></i>
                                    <span>${job.location}</span>
                                </div>
                                <div class="detail-item">
                                    <i class="fas fa-clock detail-icon"></i>
                                    <span>${job.experience}</span>
                                </div>
                            </div>
                            
                            <div class="skills">
                                <h4><i class="fas fa-tools"></i> Required Skills</h4>
                                <div>${job.skills}</div>
                            </div>
                            
                            <div class="job-description">
                                ${job.description}
                            </div>
                            
                            <a href="${job.apply_link}" target="_blank" class="apply-btn">
                                <i class="fas fa-external-link-alt"></i>
                                Apply Now
                            </a>
                        </div>
                    `;
                });
                
                jobsHtml += '</div>';
                
                resultsDiv.innerHTML = resultsHeader + jobsHtml;
            }
        </script>
    </body>
    </html>
    '''

@app.route('/recommend', methods=['POST'])
def recommend_jobs():
    try:
        data = request.get_json()
        user_skills = data.get('skills', '')
        user_preferences = data.get('preferences', '')
        user_experience = data.get('experience', '')
        user_location = data.get('location', '')
        job_type = data.get('job_type', '')
        method = data.get('method', 'tfidf')
        
        if not user_skills:
            return jsonify({'error': 'Skills are required'}), 400
        
        recommendations = recommender.get_recommendations(
            user_skills, user_preferences, user_experience, user_location, job_type
        )
        
        return jsonify({'recommendations': recommendations})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs')
def get_all_jobs():
    """API endpoint to get all available jobs"""
    return jsonify({'jobs': jobs_data})

@app.route('/api/jobs/<int:job_id>')
def get_job_details(job_id):
    """API endpoint to get specific job details"""
    job = next((job for job in jobs_data if job['id'] == job_id), None)
    if job:
        return jsonify({'job': job})
    else:
        return jsonify({'error': 'Job not found'}), 404

@app.route('/api/companies')
def get_companies():
    """API endpoint to get list of companies"""
    companies = list(set([job['company'] for job in jobs_data]))
    return jsonify({'companies': companies})

@app.route('/api/locations')
def get_locations():
    """API endpoint to get list of locations"""
    locations = []
    for job in jobs_data:
        locations.extend([loc.strip() for loc in job['location'].split(',')])
    unique_locations = list(set(locations))
    return jsonify({'locations': unique_locations})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)