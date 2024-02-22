from flask import Flask, render_template
from map_logic import run_astar_logic  # Import your A* logic function

app = Flask(__name__)

@app.route('/')
def index():
    map_data, path_data = run_astar_logic()  # Execute your A* logic function here

    return render_template('parking_lot.html', map_data=map_data, path_data=path_data)

if __name__ == '__main__':
    app.run(debug=True)
