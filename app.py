 """
Flask web application for House Plants Database
Provides web interface to browse and search plants
"""

from flask import Flask, render_template, request, jsonify
from database_queries import (
    get_all_plants, get_plant_by_id, get_plant_by_name,
    get_plants_by_category, get_all_categories, get_all_families,
    search_plants, get_low_maintenance_plants
)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/')
def index():
    """Home page with plant list"""
    plants = get_all_plants(limit=20)
    categories = get_all_categories()
    return render_template('index.html', plants=plants, categories=categories)


@app.route('/plant/<int:plant_id>')
def plant_detail(plant_id):
    """Detailed plant information page"""
    plant = get_plant_by_id(plant_id)
    if plant:
        return render_template('plant_detail.html', plant=plant)
    return "Plant not found", 404


@app.route('/category/<category>')
def category_plants(category):
    """Show all plants in a category"""
    plants = get_plants_by_category(category)
    return render_template('category.html', category=category, plants=plants)


@app.route('/search')
def search():
    """Search plants"""
    query = request.args.get('q', '')
    if query:
        results = search_plants(query)
        return render_template('search_results.html', query=query, plants=results)
    return render_template('search.html')


@app.route('/low-maintenance')
def low_maintenance():
    """Show easy care plants"""
    plants = get_low_maintenance_plants()
    return render_template('low_maintenance.html', plants=plants)


# API Endpoints
@app.route('/api/plants')
def api_plants():
    """API: Get all plants"""
    limit = request.args.get('limit', type=int)
    plants = get_all_plants(limit=limit)
    return jsonify(plants)


@app.route('/api/plant/<int:plant_id>')
def api_plant_detail(plant_id):
    """API: Get plant by ID"""
    plant = get_plant_by_id(plant_id)
    if plant:
        return jsonify(plant)
    return jsonify({'error': 'Plant not found'}), 404


@app.route('/api/search')
def api_search():
    """API: Search plants"""
    query = request.args.get('q', '')
    if query:
        results = search_plants(query)
        return jsonify(results)
    return jsonify([])


@app.route('/api/categories')
def api_categories():
    """API: Get all categories"""
    categories = get_all_categories()
    return jsonify(categories)


if __name__ == '__main__':
    app.run(debug=True, port=5000)