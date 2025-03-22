from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from scipy.optimize import minimize

app = Flask(__name__)
CORS(app)  
CORS(app, origins="*")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///financial_literacy.db'  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class User(db.Model):
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    experience_points = db.Column(db.Integer, default=0)
    credit_score = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    location = db.Column(db.String(100), nullable=False)
    salary = db.Column(db.Numeric(10, 2), default=0)


class LearningModule(db.Model):
    module_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    xp_award = db.Column(db.Integer, nullable=False)

class UserProgress(db.Model):
    progress_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    module_id = db.Column(db.Integer, db.ForeignKey('learning_module.module_id'), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.DateTime)


class Goal(db.Model):
    goal_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    goal_name = db.Column(db.String(100), nullable=False)
    year_of_completion = db.Column(db.Integer, nullable=False)
    amount = db.Column(db.Numeric(10, 2), nullable=False)


class LivingCost(db.Model):
    location_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    location_name = db.Column(db.String(100), unique=True, nullable=False)
    average_cost = db.Column(db.Numeric(10, 2), nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)


class SalaryTransaction(db.Model):
    transaction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    type = db.Column(db.Enum('Earning', 'Deduction'), nullable=False)
    description = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()  

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    new_user = User(username=data['username'], password=hashed_password, email=data['email'], location=data['location'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()

    if user and check_password_hash(user.password, data['password']):
       
        user_goals = Goal.query.filter_by(user_id=user.user_id).all()
        goals_list = [{
            'goal_name': goal.goal_name,
            'year_of_completion': goal.year_of_completion,
            'amount': float(goal.amount),
            
        } for goal in user_goals]

        return jsonify({
            'message': 'Login successful',
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'location': user.location,
            'experience_points': user.experience_points,
            'credit_score': user.credit_score,
            'salary': float(user.salary),
            'goals_list': goals_list
        }), 200
    
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    users = User.query.order_by(User.experience_points.desc()).all()
    
    leaderboard_data = [{
        'rank': index + 1,
        'username': user.username,
        'experience_points': user.experience_points
    } for index, user in enumerate(users)]
    
    return jsonify({'leaderboard': leaderboard_data}), 200

@app.route('/profile', methods=['GET'])
def profile():
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'username': user.username,
        'email': user.email,
        'experience_points': user.experience_points,
        'credit_score': user.credit_score,
        'location': user.location,
        'salary': float(user.salary)
    }), 200


@app.route('/fetch_goals', methods=['GET'])
def fetch_goals():
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    user_goals = Goal.query.filter_by(user_id=user_id).all()
    if not user_goals:
        return jsonify({'message': 'No goals found'}), 404

    goals_list = [{
        'goal_id': goal.goal_id,
        'goal_name': goal.goal_name,
        'year_of_completion': goal.year_of_completion,
        'amount': float(goal.amount),
        
    } for goal in user_goals]

    return jsonify({'goals': goals_list}), 200
@app.route('/update_experience', methods=['POST'])
def update_experience():
    data = request.json
    user = User.query.filter_by(user_id=data['user_id']).first()
    
    if user:
        user.experience_points += data['points']
        db.session.commit()
        return jsonify({'message': 'Experience points updated successfully'}), 200
    
    return jsonify({'message': 'User not found'}), 404
@app.route('/add_goal', methods=['POST'])
def add_goal():
    data = request.json

    
    required_fields = ['user_id', 'goal_name', 'year_of_completion', 'amount']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    
    new_goal = Goal(
        user_id=data['user_id'],
        goal_name=data['goal_name'],
        year_of_completion=data['year_of_completion'],
        amount=data['amount'],
        
    )
    
    db.session.add(new_goal)
    db.session.commit()

    return jsonify({'message': 'Goal added successfully'}), 201

def fetch_inflation_rate_cpi():
    try:
        start_date = datetime(2010, 1, 1)
        end_date = datetime.today()
        inflation_data = pdr.DataReader("FPCPITOTLZGIND", "fred", start_date, end_date)
        latest_inflation = inflation_data.iloc[-1, 0]
        return latest_inflation
    except Exception:
        return 5.0 


def optimize_portfolio(data,riskFreeRate):
    trading_days = 252  
    
    annual_returns = ((1 + data.pct_change(fill_method=None).mean()) ** trading_days) - 1
    returns_cov = data.pct_change(fill_method=None).cov() * trading_days 
    
    
    print("Annualized Returns:\n", annual_returns)
    print("Covariance Matrix:\n", returns_cov)

    risk_free_rate = riskFreeRate

    def objective(weights):
        portfolio_return = np.dot(weights, annual_returns) 
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns_cov, weights)))  

        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        
        
        penalty = np.sum(weights**2)  
        
        return -sharpe_ratio + penalty 

    
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    
   
    bounds = [(0, 1) for _ in range(len(annual_returns))]
    
   
    initial_weights = np.ones(len(annual_returns)) / len(annual_returns)

    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    print("Optimization result:", result)

   
    return result.x if result.success else initial_weights


def calculate_future_value(monthly_investment, growth_rate, years, weights, returns):
    portfolio_values = np.zeros(len(weights))  
    annual_investment = 12 * monthly_investment
    accumulated_value = np.zeros(len(weights))  

    for year in range(1, years + 1):
        
        annual_investment = annual_investment * (1 + growth_rate / 100)
        portfolio_values = np.zeros(len(weights))  

       
        portfolio_values += weights * annual_investment

      
        portfolio_values += accumulated_value

       
        yearly_return = np.dot(weights, returns)
        portfolio_values *= (1 + yearly_return)  

       
        accumulated_value = portfolio_values.copy()

    total_value = portfolio_values.sum()  
    return portfolio_values, total_value


def jsonify_results(results):
    for goal_status in results["goals_status"]:
        goal_status["achieved"] = bool(goal_status["achieved"]) 
    return jsonify(results)


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    monthly_investment = data['monthly_investment']
    growth_rate = data['growth_rate']
    goals = data['goals']
    riskFreeRate=data['riskFreeRate']
    print('*****************************************')
    print('Risk Free Rate: =',riskFreeRate)

    tickers = ["^NSEI", "^BSESN", "GLD", "0P0001BB7Q.BO"]
    start_date = "2010-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    stock_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    weights = optimize_portfolio(stock_data,riskFreeRate)
    trading_days = 252  
    if stock_data.empty:
     print("error No stock data available")  # Handle gracefully

    returns = ((1 + stock_data.pct_change(fill_method=None)).prod() ** (trading_days / len(stock_data))) -1
    returns = returns[::-1]

    
    inflation_rate = fetch_inflation_rate_cpi()

    results = {"goals_status": [], "optimal_weights": dict(zip(tickers, weights))}

    print(tickers,weights,returns)

    for goal in goals:
        target = goal['target']
        years = goal['years']
        inflation_adjusted_target = target * ((1 + inflation_rate / 100) ** years)
        portfolio_values, total_value = calculate_future_value(monthly_investment, growth_rate, years, weights, returns)
        print(portfolio_values)

        results["goals_status"].append({
            "goal": goal,
            "achieved": total_value >= inflation_adjusted_target,
            "future_value": total_value,
            "inflation_adjusted_target": inflation_adjusted_target
        })
    
    return jsonify_results(results)




if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=False)

