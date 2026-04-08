from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'mental_health_secret_key_2024'

# Global variables
model = None
label_encoder = None
feature_names = None
model_accuracy = None
clf_report = None
sentiment_analyzer = None
question_columns = None  # For Likert 1-5 form-driven dataset

def load_and_train_model():
    """Load dataset and train the ML model"""
    global model, label_encoder, feature_names, model_accuracy, clf_report, sentiment_analyzer, question_columns
    
    # Prefer the shared form-driven dataset if present (Likert 1-5 scale)
    form_csv = 'Mental Health Emotion Survey- AI research (Responses) - Form Responses 1.csv'
    if os.path.exists(form_csv):
        (
            model,
            label_encoder,
            feature_names,
            model_accuracy,
            clf_report,
            question_columns,
        ) = train_from_form_csv(form_csv)
    else:
        (
            model,
            label_encoder,
            feature_names,
            model_accuracy,
            clf_report,
        ) = train_from_binary_csv('Mental_health_dset.csv')
    

    # Initialize sentiment analyzer (download once)
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    sentiment_analyzer = SentimentIntensityAnalyzer()

    print(f"Model trained and saved successfully! Holdout accuracy: {model_accuracy:.4f}")

def train_from_binary_csv(csv_path):
    """Train model on original binary yes/no dataset."""
    df = pd.read_csv(csv_path)
    feature_columns = [c for c in df.columns if c != 'Disorder']
    X = df[feature_columns].apply(lambda x: (x == 'yes').astype(int))
    y = df['Disorder']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    eval_model = RandomForestClassifier(
        n_estimators=400, class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
    eval_model.fit(X_train, y_train)
    acc = float(eval_model.score(X_test, y_test))
    rep = classification_report(y_test, eval_model.predict(X_test), output_dict=True)

    final_model = RandomForestClassifier(
        n_estimators=600, class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
    final_model.fit(X, y_enc)
    joblib.dump(final_model, 'mental_health_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    return final_model, le, feature_columns, acc, rep

def train_from_form_csv(csv_path):
    """Train model on Likert-scale form responses (1-5)."""
    df = pd.read_csv(csv_path)
    # Identify question columns between demographics and 'Disorder'
    # Assume last column is 'Disorder'
    cols = list(df.columns)
    if 'Disorder' in cols:
        end_idx = cols.index('Disorder')
    else:
        end_idx = len(cols)
    # Skip first 5 demographic columns (Timestamp, Age, Gender, Occupation, Locality)
    start_idx = 5
    q_cols = cols[start_idx:end_idx]
    # Keep up to first 20, but not zero
    q_cols = q_cols[:20]
    if len(q_cols) == 0:
        # Fallback to binary dataset if no questions detected
        return train_from_binary_csv('Mental_health_dset.csv') + (q_cols,)

    # Build numeric feature matrix and impute missing with column medians (default 3)
    X_full = df[q_cols].apply(pd.to_numeric, errors='coerce')
    y_full = df['Disorder'] if 'Disorder' in df.columns else None
    # Drop rows without label only if y exists
    if y_full is not None:
        mask = y_full.notnull()
        X_full = X_full[mask]
        y_full = y_full[mask]
    # Impute
    medians = X_full.median(numeric_only=True)
    medians = medians.fillna(3)
    X = X_full.fillna(medians)
    y = y_full if y_full is not None else None

    # If labels are present, supervised training
    if y is not None:
        if len(X) == 0 or len(y) == 0:
            # No clean samples after preprocessing: fallback to binary dataset
            fm, le, feats, acc, rep = train_from_binary_csv('Mental_health_dset.csv')
            # When falling back, we are not in Likert mode
            return fm, le, feats, acc, rep, None
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        # Safe split: handle small class counts and ensure non-empty splits
        def _safe_split(Xm, ym):
            n = len(Xm)
            if n < 3:
                return Xm, Xm, ym, ym
            # Check per-class counts for stratify suitability
            unique, counts = np.unique(ym, return_counts=True)
            if counts.min() >= 2 and len(unique) > 1:
                ts = 0.2 if int(n * 0.2) >= 1 else 1 / n
                return train_test_split(Xm, ym, test_size=ts, random_state=42, stratify=ym)
            else:
                ts = 0.2 if int(n * 0.2) >= 1 else 1 / n
                return train_test_split(Xm, ym, test_size=ts, random_state=42)

        X_train, X_test, y_train, y_test = _safe_split(X, y_enc)
        eval_model = RandomForestClassifier(
            n_estimators=500, class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )
        if len(X_train) == 0 or len(y_train) == 0:
            # Edge case: split produced empty train
            X_train, y_train = X, y_enc
            X_test, y_test = X, y_enc
        eval_model.fit(X_train, y_train)
        acc = float(eval_model.score(X_test, y_test))
        rep = classification_report(y_test, eval_model.predict(X_test), output_dict=True)

        final_model = RandomForestClassifier(
            n_estimators=800, class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )
        final_model.fit(X, y_enc)
        joblib.dump(final_model, 'mental_health_model.pkl')
        joblib.dump(le, 'label_encoder.pkl')
        return final_model, le, q_cols, acc, rep, q_cols
    else:
        # No labels: fall back to original binary dataset and disable Likert mode
        fm, le, feats, acc, rep = train_from_binary_csv('Mental_health_dset.csv')
        return fm, le, feats, acc, rep, None

def predict_disorder(form_responses):
    """Predict mental health disorder based on form responses"""
    if model is None:
        load_and_train_model()
    
    # Build feature vector depending on training mode
    features = []
    if question_columns:
        # Likert (1-5) mode: accept numeric strings; default to 3 if missing
        for i, q in enumerate(question_columns):
            val = form_responses.get(f'q{i+1}', form_responses.get(q, '3'))
            try:
                v = int(val)
            except (TypeError, ValueError):
                v = 3
            features.append(v)
    else:
        # Binary yes/no mode
        for feature in feature_names:
            if feature in form_responses:
                features.append(1 if str(form_responses[feature]).lower() == 'yes' else 0)
            else:
                features.append(0)
    
    # Align feature length with trained model expectations
    try:
        prediction = model.predict([features])[0]
    except ValueError as ve:
        # If mismatch (e.g., trained on 24 but got 20), pad/truncate accordingly
        n_expected = getattr(model, 'n_features_in_', None)
        if n_expected is None:
            raise
        adj = list(features)
        if len(adj) < n_expected:
            adj = adj + [3] * (n_expected - len(adj))  # neutral padding for Likert
        elif len(adj) > n_expected:
            adj = adj[:n_expected]
        prediction = model.predict([adj])[0]
    disorder = label_encoder.inverse_transform([prediction])[0]
    
    # Calculate confidence score (1-5 scale)
    confidence_scores = model.predict_proba([features])[0]
    max_confidence = max(confidence_scores)
    severity_scale = max(1, min(5, int(round(max_confidence * 5))))
    
    return disorder, severity_scale, max_confidence

@app.route('/')
def home():
    """Home page with option to take test"""
    return render_template('index.html')

@app.route('/take_test')
def take_test():
    """Render in-site assessment test page (Likert questions if available)."""
    if question_columns:
        questions = [{'id': i+1, 'text': question_columns[i]} for i in range(len(question_columns))]
        scale = [1,2,3,4,5]
        return render_template('test.html', questions=questions, scale=scale, likert=True)
    return render_template('test.html', questions=None, scale=[1,2,3,4,5], likert=False)

@app.route('/process_test', methods=['POST'])
def process_test():
    """Process test results and start chatbot session"""
    try:
        # Get form data from the test
        form_data = request.json
        
        # Predict disorder
        disorder, severity, confidence = predict_disorder(form_data)
        
        # Store in session for chatbot
        session['test_results'] = {
            'disorder': disorder,
            'severity': severity,
            'confidence': confidence,
            'form_data': form_data
        }
        
        # Initialize chatbot session
        session['chatbot_questions'] = get_chatbot_questions(disorder)
        session['chatbot_answers'] = {}
        session['current_question'] = 0
        
        return jsonify({
            'success': True,
            'disorder': disorder,
            'severity': severity,
            'redirect': url_for('chatbot')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chatbot')
def chatbot():
    """Chatbot interface for follow-up questions"""
    if 'test_results' not in session:
        return redirect(url_for('home'))
    
    return render_template('chatbot.html')

@app.route('/chatbot/next_question', methods=['POST'])
def next_question():
    """Get next chatbot question"""
    if 'chatbot_questions' not in session:
        return jsonify({'error': 'No questions available'})
    
    current_q = session.get('current_question', 0)
    questions = session['chatbot_questions']
    
    if current_q >= len(questions):
        return jsonify({'completed': True})
    
    question = questions[current_q]
    return jsonify({
        'question': question['text'],
        'question_number': current_q + 1,
        'total_questions': len(questions)
    })

@app.route('/chatbot/answer', methods=['POST'])
def submit_answer():
    """Submit answer to chatbot question"""
    data = request.json
    answer = data.get('answer', '')
    # Ensure question_id is an integer for arithmetic, but store as string in session
    try:
        question_id = int(data.get('question_id', 0))
    except (TypeError, ValueError):
        question_id = 0
    
    if 'chatbot_answers' not in session:
        session['chatbot_answers'] = {}
    
    # Use string keys to avoid JSON serialization issues in Flask sessions
    session['chatbot_answers'][str(question_id)] = answer
    session['current_question'] = question_id + 1
    
    # Check if all questions answered
    questions = session.get('chatbot_questions', [])
    if session['current_question'] >= len(questions):
        return jsonify({'completed': True, 'redirect': url_for('generate_report')})
    
    return jsonify({'success': True, 'next_question': True})

@app.route('/generate_report')
def generate_report():
    """Generate comprehensive mental health report"""
    if 'test_results' not in session or 'chatbot_answers' not in session:
        return redirect(url_for('home'))
    
    test_results = session['test_results']
    chatbot_answers = session['chatbot_answers']
    
    # Generate report
    report = generate_mental_health_report(test_results, chatbot_answers)
    
    return render_template('report.html', report=report)

def get_chatbot_questions(disorder):
    """Get relevant follow-up questions based on predicted disorder"""
    questions_map = {
        'Anxiety': [
            {'id': 0, 'text': 'What specific situations make you feel most anxious?'},
            {'id': 1, 'text': 'How do you typically cope with anxiety when it occurs?'},
            {'id': 2, 'text': 'What physical symptoms do you experience during anxiety?'},
            {'id': 3, 'text': 'How long do these anxious feelings typically last?'},
            {'id': 4, 'text': 'What would help you feel more calm and secure?'}
        ],
        'Depression': [
            {'id': 0, 'text': 'What activities or hobbies used to bring you joy?'},
            {'id': 1, 'text': 'How has your sleep pattern changed recently?'},
            {'id': 2, 'text': 'What thoughts go through your mind when you feel low?'},
            {'id': 3, 'text': 'How do you motivate yourself on difficult days?'},
            {'id': 4, 'text': 'What support do you feel you need most right now?'}
        ],
        'Stress': [
            {'id': 0, 'text': 'What are the main sources of stress in your life?'},
            {'id': 1, 'text': 'How do you currently manage stress?'},
            {'id': 2, 'text': 'What would help you feel more in control?'},
            {'id': 3, 'text': 'How does stress affect your daily routine?'},
            {'id': 4, 'text': 'What relaxation techniques have you tried?'}
        ],
        'Loneliness': [
            {'id': 0, 'text': 'What kind of social connections do you desire?'},
            {'id': 1, 'text': 'What prevents you from reaching out to others?'},
            {'id': 2, 'text': 'How do you spend your free time?'},
            {'id': 3, 'text': 'What activities would help you meet new people?'},
            {'id': 4, 'text': 'What support do you need to build connections?'}
        ],
        'Normal': [
            {'id': 0, 'text': 'How do you maintain your mental well-being?'},
            {'id': 1, 'text': 'What coping strategies work best for you?'},
            {'id': 2, 'text': 'How do you recognize when you need support?'},
            {'id': 3, 'text': 'What healthy habits do you practice?'},
            {'id': 4, 'text': 'How do you support others with their mental health?'}
        ]
    }
    
    return questions_map.get(disorder, questions_map['Normal'])

def generate_mental_health_report(test_results, chatbot_answers):
    """Generate comprehensive mental health report"""
    disorder = test_results['disorder']
    severity = test_results['severity']
    
    # Get recommendations and resources
    recommendations = get_recommendations(disorder, severity)
    resources = get_resources(disorder)
    
    # Heuristic severity adjustment using symptom count and chatbot sentiment
    symptom_signal = compute_symptom_signal(test_results.get('form_data', {}))
    sentiment_signal = analyze_sentiment_signal(chatbot_answers)
    adjusted_severity = adjust_severity(severity, symptom_signal, sentiment_signal)

    report = {
        'disorder': disorder,
        'severity_scale': adjusted_severity,
        'severity_description': get_severity_description(adjusted_severity),
        'recommendations': recommendations,
        'resources': resources,
        'chatbot_insights': analyze_chatbot_answers(chatbot_answers, disorder),
        'generated_date': datetime.now().strftime('%B %d, %Y'),
        'next_steps': get_next_steps(disorder, adjusted_severity),
        'model_accuracy': model_accuracy,
        'per_class_metrics': summarize_classification_report(clf_report),
        'signals': {
            'symptom_signal': symptom_signal,
            'sentiment_signal': sentiment_signal
        },
        'question_mode': 'likert' if question_columns else 'binary'
    }
    
    return report

def get_severity_description(severity):
    """Get description for severity scale"""
    descriptions = {
        1: "Very Mild - Minimal impact on daily life",
        2: "Mild - Some impact but manageable",
        3: "Moderate - Noticeable impact on daily activities",
        4: "Severe - Significant impact on daily life",
        5: "Very Severe - Major impact requiring immediate attention"
    }
    return descriptions.get(severity, "Unknown severity level")

def get_recommendations(disorder, severity):
    """Get personalized recommendations based on disorder and severity"""
    base_recommendations = {
        'Anxiety': [
            "Practice deep breathing exercises (4-7-8 technique)",
            "Use progressive muscle relaxation",
            "Limit caffeine and alcohol intake",
            "Establish a regular sleep schedule",
            "Consider mindfulness meditation"
        ],
        'Depression': [
            "Maintain a regular daily routine",
            "Exercise regularly (even light walking)",
            "Connect with supportive friends/family",
            "Practice self-compassion",
            "Consider journaling your thoughts"
        ],
        'Stress': [
            "Practice time management techniques",
            "Learn to say 'no' to excessive commitments",
            "Engage in regular physical activity",
            "Practice stress-reduction techniques",
            "Maintain work-life balance"
        ],
        'Loneliness': [
            "Join community groups or clubs",
            "Volunteer for causes you care about",
            "Reach out to old friends",
            "Consider pet companionship",
            "Engage in social hobbies"
        ],
        'Normal': [
            "Continue maintaining healthy habits",
            "Support others in their mental health journey",
            "Stay connected with your support network",
            "Practice gratitude and mindfulness",
            "Regular mental health check-ins"
        ]
    }
    
    recommendations = base_recommendations.get(disorder, base_recommendations['Normal'])
    
    # Add severity-specific recommendations
    if severity >= 4:
        recommendations.insert(0, "Consider seeking professional mental health support")
        recommendations.insert(1, "Talk to a trusted healthcare provider")
    
    return recommendations

def get_resources(disorder):
    """Get relevant resources and contacts"""
    resources = {
        'emergency': {
            'name': 'Crisis Helpline',
            'phone': '7259207798',
            'description': '24/7 Suicide & Crisis Lifeline'
        },
        'general': {
            'name': 'Mental Health Care India',
            'phone': '7259207798',
            'website': 'https://telemanas.mohfw.gov.in/home'
        },
        'disorder_specific': {}
    }
    
    # Add disorder-specific resources
    if disorder == 'Anxiety':
        resources['disorder_specific'] = {
            'name': 'Anxiety and Depression Association of India',
            'website': 'https://telemanas.mohfw.gov.in/home',
            'phone': '7259207798'
        }
    elif disorder == 'Depression':
        resources['disorder_specific'] = {
            'name': 'Depression and Bipolar Support Alliance',
            'website': 'https://telemanas.mohfw.gov.in/home',
            'phone': '7259207798'
        }
    
    return resources

def analyze_chatbot_answers(chatbot_answers, disorder):
    """Analyze chatbot answers for insights"""
    insights = []
    
    if chatbot_answers:
        insights.append(f"Based on your responses, you've provided valuable insights about your experience with {disorder.lower()}.")
        
        # Add specific insights based on answers
        if len(chatbot_answers) >= 3:
            insights.append("Your detailed responses show self-awareness and willingness to work on your mental health.")
        
        insights.append("Consider discussing these insights with a mental health professional for personalized guidance.")
    
    return insights

def analyze_sentiment_signal(chatbot_answers):
    """Aggregate sentiment of chatbot answers into a 0-1 severity signal."""
    if not chatbot_answers or not sentiment_analyzer:
        return 0.0
    texts = [str(v) for k, v in sorted(chatbot_answers.items(), key=lambda kv: str(kv[0]))]
    if not texts:
        return 0.0
    scores = [sentiment_analyzer.polarity_scores(t)['compound'] for t in texts]
    # Map VADER compound (-1..1) to severity 0..1 (more negative -> higher severity)
    neg = [(1 - (s + 1) / 2) for s in scores]
    return float(min(1.0, max(0.0, sum(neg) / len(neg))))

def compute_symptom_signal(form_data):
    """Compute fraction of 'yes' answers across known features."""
    if not form_data:
        return 0.0
    yes_count = 0
    total = 0
    for feature in feature_names or []:
        if feature in form_data:
            total += 1
            yes_count += 1 if str(form_data[feature]).lower() == 'yes' else 0
    if total == 0:
        return 0.0
    return float(yes_count / total)

def adjust_severity(model_severity, symptom_signal, sentiment_signal):
    """Combine model confidence-derived severity with symptom and sentiment signals."""
    base = model_severity
    # Weighted adjustment: symptoms 50%, sentiment 30%, model 20%
    combined = 0.2 * (base / 5.0) + 0.5 * symptom_signal + 0.3 * sentiment_signal
    scaled = int(round(max(1.0, min(5.0, combined * 5.0))))
    return scaled

def summarize_classification_report(report_dict):
    if not isinstance(report_dict, dict):
        return None
    summary = {}
    for label, metrics in report_dict.items():
        if label in ('accuracy', 'macro avg', 'weighted avg'):
            summary[label] = metrics if isinstance(metrics, float) else metrics.get('f1-score')
    return summary

def get_next_steps(disorder, severity):
    """Get recommended next steps"""
    steps = []
    
    if severity >= 4:
        steps.extend([
            "Schedule an appointment with a mental health professional",
            "Consider reaching out to a crisis helpline if needed",
            "Inform a trusted friend or family member about your situation"
        ])
    elif severity >= 3:
        steps.extend([
            "Consider consulting with a mental health professional",
            "Implement the recommended self-help strategies",
            "Monitor your symptoms and track your progress"
        ])
    else:
        steps.extend([
            "Continue with the recommended self-help strategies",
            "Maintain regular mental health check-ins",
            "Consider preventive mental health care"
        ])
    
    steps.append("Follow up with your healthcare provider if symptoms persist or worsen")
    
    return steps

if __name__ == '__main__':
    # Load and train model on startup
    print("Loading and training mental health prediction model...")
    load_and_train_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)

