from app import db, app  # Ensure you import both db and app

# Create an application context
with app.app_context():
    db.drop_all()  # Deletes all tables
    db.create_all()  # Recreates fresh tables
    print("All tables dropped and recreated successfully.")
