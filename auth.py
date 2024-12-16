# auth.py
users_db = {} 

def register_user(first_name, last_name, email, password):
    if email in users_db:
        return {'status': 'error', 'message': 'Email already registered'}
    
    users_db[email] = {'first_name': first_name, 'last_name': last_name, 'password': password}
    return {'status': 'success', 'message': 'Registration successful'}

def authenticate_user(email, password):
    user = users_db.get(email)
    if user and user['password'] == password:
        return {'status': 'success', 'message': 'Login successful'}
    return {'status': 'error', 'message': 'Invalid email or password'}
