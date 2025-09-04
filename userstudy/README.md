# UserStudy

A Flask-based web application for conducting user studies for TAC and MAC models.

## Requirements

- Python 3.10 or higher
- Flask
- Gunicorn (for production deployment)
- Additional dependencies needed for running swift framework

## Installation

### Local Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd userstudy
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: The requirements.txt file includes only the basic dependencies. You may need to install additional packages depending on which models you want to use:


3. Prepare model checkpoints:
   - Ensure the `ckpt` directory contains the necessary model files for Query Blazer (QB) and MPC models

## Running the Application

### Running Locally

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. The application will be available at:
   ```
   http://localhost:5000/
   ```

3. To start a new session, navigate to:
   ```
   http://localhost:5000/?new_example=true
   ```

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t userstudy .
   ```

2. Run the container:
   ```bash
   docker run -p 8080:8080 userstudy
   ```

3. The application will be available at:
   ```
   http://localhost:8080/
   ```

## Application Structure

- `app.py`: Main application file containing the Flask server and model implementations
- `conv.py`: Contains conversation pools and utilities
- `models/`: Directory containing model implementations
- `static/`: Frontend files (HTML, CSS, JavaScript)
- `imagechat_samples/`: Sample data for image-based conversations

## Usage

1. When the application starts, it randomly selects a conversation context from the available pool
2. The MiniCPM model is used by default for text completion
3. The application logs user interactions in `session_logs.csv` and context information in `context_log.csv`

