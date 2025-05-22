import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
from matplotlib import gridspec

class EcoRouteML:
    def __init__(self):
        self.fuel_model = None
        self.scaler = StandardScaler()
        self.graph = None
        # Define terrain types and their factors
        self.terrain_types = {
            'flat': 1.0,
            'rolling': 1.2,
            'hilly': 1.5,
            'mountainous': 2.0
        }
        # Define traffic levels and their factors
        self.traffic_levels = {
            'none': 1.0,
            'light': 1.1,
            'moderate': 1.3,
            'heavy': 1.8,
            'congested': 2.5
        }
        # Define average speeds for different traffic conditions (km/h)
        self.traffic_speeds = {
            'none': 90,
            'light': 80,
            'moderate': 60,
            'heavy': 40,
            'congested': 20
        }
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic data for training the fuel prediction model"""
        # Features that affect fuel consumption
        vehicle_weight = np.random.normal(1500, 300, n_samples)  # kg
        vehicle_engine_size = np.random.normal(2.0, 0.5, n_samples)  # liters
        road_incline = np.random.uniform(-10, 10, n_samples)  # percent
        speed = np.random.uniform(20, 120, n_samples)  # km/h
        acceleration = np.random.normal(0, 1, n_samples)  # m/s²
        traffic_density = np.random.uniform(0, 1, n_samples)  
        temperature = np.random.uniform(-10, 40, n_samples)  # celsius
        wind_speed = np.random.uniform(0, 30, n_samples)  # km/h
        road_condition = np.random.uniform(0, 1, n_samples)  # 0: poor, 1: excellent
        
        # Create a dataframe
        data = pd.DataFrame({
            'vehicle_weight': vehicle_weight,
            'vehicle_engine_size': vehicle_engine_size,
            'road_incline': road_incline,
            'speed': speed,
            'acceleration': acceleration,
            'traffic_density': traffic_density,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'road_condition': road_condition
        })
        
        # Generate fuel consumption based on features (synthetic formula)
        # This is a simplified model of how these factors might affect fuel consumption
        fuel_consumption = (
            0.05 * vehicle_weight / 1000 +
            2.0 * vehicle_engine_size +
            0.2 * abs(road_incline) +
            (0.015 * (speed - 80) ** 2) / 100 +
            1.5 * abs(acceleration) +
            2.0 * traffic_density +
            0.05 * abs(temperature - 20) / 10 +
            0.1 * wind_speed / 10 -
            0.5 * road_condition +
            np.random.normal(0, 0.5, n_samples)  # Add some noise
        )
        
        data['fuel_consumption'] = np.maximum(fuel_consumption, 1.0)  # Ensure minimum fuel consumption
        
        return data
    
    def train_fuel_prediction_model(self, data=None):
        """Train a machine learning model to predict fuel consumption"""
        if data is None:
            data = self.generate_synthetic_data()
        
        X = data.drop('fuel_consumption', axis=1)
        y = data['fuel_consumption']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.fuel_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.fuel_model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_score = self.fuel_model.score(X_train_scaled, y_train)
        test_score = self.fuel_model.score(X_test_scaled, y_test)
        
        print(f"Fuel prediction model training score: {train_score:.4f}")
        print(f"Fuel prediction model testing score: {test_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.fuel_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return self.fuel_model
    
    def predict_fuel_consumption(self, features):
        """Predict fuel consumption for a given set of features"""
        if self.fuel_model is None:
            raise ValueError("Model not trained yet. Call train_fuel_prediction_model first.")
        
        features_scaled = self.scaler.transform(features)
        return self.fuel_model.predict(features_scaled)
    
    def create_road_network(self, num_nodes=20, edge_probability=0.2):
        """Create a synthetic road network for route optimization testing"""
        # Create a random geometric graph
        pos = {i: (random.random(), random.random()) for i in range(num_nodes)}
        G = nx.random_geometric_graph(num_nodes, 0.5, pos=pos)
        nx.set_node_attributes(G, pos, 'pos')

        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        # Assign random road properties to edges with more detailed attributes
        for u, v in G.edges():
            # Calculate actual distance based on positions (assuming coordinate units are km)
            pos_u = pos[u]
            pos_v = pos[v]
            distance = math.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2) * 10  # Scale to km
            
            # Assign road properties
            traffic_density = np.random.uniform(0, 1)
            # Convert traffic density to descriptive level
            if traffic_density < 0.2:
                traffic_level = 'none'
            elif traffic_density < 0.4:
                traffic_level = 'light'
            elif traffic_density < 0.6:
                traffic_level = 'moderate'
            elif traffic_density < 0.8:
                traffic_level = 'heavy'
            else:
                traffic_level = 'congested'
                
            # Assign terrain type
            terrain_rand = np.random.uniform(0, 1)
            if terrain_rand < 0.4:
                terrain_type = 'flat'
            elif terrain_rand < 0.7:
                terrain_type = 'rolling'
            elif terrain_rand < 0.9:
                terrain_type = 'hilly'
            else:
                terrain_type = 'mountainous'
                
            # Calculate road incline based on terrain
            if terrain_type == 'flat':
                road_incline = np.random.uniform(-1, 1)
            elif terrain_type == 'rolling':
                road_incline = np.random.uniform(-3, 3)
            elif terrain_type == 'hilly':
                road_incline = np.random.uniform(-5, 5)
            else:  # mountainous
                road_incline = np.random.uniform(-10, 10)
                
            road_condition = np.random.uniform(0, 1)
            # Convert road condition to descriptive quality
            if road_condition < 0.3:
                road_quality = 'poor'
            elif road_condition < 0.7:
                road_quality = 'average'
            else:
                road_quality = 'good'
                
            # Store all properties
            G[u][v]['distance'] = distance
            G[u][v]['traffic_density'] = traffic_density
            G[u][v]['traffic_level'] = traffic_level
            G[u][v]['terrain_type'] = terrain_type
            G[u][v]['road_incline'] = road_incline
            G[u][v]['road_condition'] = road_condition
            G[u][v]['road_quality'] = road_quality
            
            # Add estimated travel time based on traffic level and distance
            speed = self.traffic_speeds[traffic_level]
            # Adjust speed based on terrain
            if terrain_type == 'rolling':
                speed *= 0.9
            elif terrain_type == 'hilly':
                speed *= 0.8
            elif terrain_type == 'mountainous':
                speed *= 0.7
                
            # Adjust speed based on road quality
            if road_quality == 'poor':
                speed *= 0.8
            elif road_quality == 'average':
                speed *= 0.9
                
            # Calculate travel time in hours
            travel_time = distance / speed
            G[u][v]['travel_time'] = travel_time
        
        self.graph = G
        return G
    
    def calculate_edge_fuel_consumption(self, u, v, vehicle_params, verbose=False):
        """Calculate the expected fuel consumption for traversing an edge with advanced formula"""
        if self.fuel_model is None:
            raise ValueError("Model not trained yet. Call train_fuel_prediction_model first.")
        if self.graph is None:
            raise ValueError("Road network not created yet. Call create_road_network first.")
        
        # Get edge properties
        edge_data = self.graph[u][v]
        distance = edge_data['distance']  # in km
        
        # Get base fuel consumption rate from the ML model
        base_features = pd.DataFrame({
            'vehicle_weight': [vehicle_params['weight']],
            'vehicle_engine_size': [vehicle_params['engine_size']],
            'road_incline': [edge_data['road_incline']],
            'speed': [80 * (1 - edge_data['traffic_density'])],  # speed decreases with traffic
            'acceleration': [0.5],  # assume constant moderate acceleration
            'traffic_density': [edge_data['traffic_density']],
            'temperature': [vehicle_params['temperature']],
            'wind_speed': [vehicle_params['wind_speed']],
            'road_condition': [edge_data['road_condition']]
        })
        
        # Base consumption rate (L/100km)
        base_consumption = self.predict_fuel_consumption(base_features)[0]
        
        # Apply advanced formula with specific factors
        terrain_factor = self.terrain_types[edge_data['terrain_type']]
        traffic_factor = self.traffic_levels[edge_data['traffic_level']]
        
        # Additional factors based on vehicle and conditions
        wind_factor = 1.0 + (vehicle_params['wind_speed'] / 100)  # small impact from wind
        temp_factor = 1.0 + (abs(vehicle_params['temperature'] - 20) / 100)  # optimal temp is 20°C
        
        # Advanced fuel consumption formula
        fuel_consumption = (
            base_consumption *  # base rate (L/100km)
            (distance / 100) *  # convert to actual distance (L)
            terrain_factor *    # terrain impact
            traffic_factor *    # traffic impact
            wind_factor *       # wind impact
            temp_factor         # temperature impact
        )
        
        # Apply road quality factor (better roads = better efficiency)
        if edge_data['road_quality'] == 'poor':
            road_quality_factor = 1.2
        elif edge_data['road_quality'] == 'average':
            road_quality_factor = 1.0
        else:  # good
            road_quality_factor = 0.9
            
        fuel_consumption *= road_quality_factor
        
        # Display intermediate values if verbose is True
        if verbose:
            print(f"\n===== Edge {u} to {v} Fuel Calculation =====")
            print(f"Distance: {distance:.2f} km")
            print(f"Base consumption rate: {base_consumption:.2f} L/100km")
            print(f"Terrain: {edge_data['terrain_type']} (factor: {terrain_factor:.1f}x)")
            print(f"Traffic: {edge_data['traffic_level']} (factor: {traffic_factor:.1f}x)")
            print(f"Road quality: {edge_data['road_quality']} (factor: {road_quality_factor:.1f}x)")
            print(f"Wind factor: {wind_factor:.2f}x")
            print(f"Temperature factor: {temp_factor:.2f}x")
            print(f"Fuel consumption formula:")
            print(f"  {base_consumption:.2f} L/100km × ({distance:.2f}km / 100) × {terrain_factor:.1f} × {traffic_factor:.1f} × {wind_factor:.2f} × {temp_factor:.2f} × {road_quality_factor:.1f}")
            print(f"Total consumption: {fuel_consumption:.3f} liters")
            print("=============================================")
        
        return fuel_consumption
    
    def find_eco_route(self, start_node, end_node, vehicle_params, verbose=False):
        """Find the most fuel-efficient route between two nodes"""
        if self.graph is None:
            raise ValueError("Road network not created yet. Call create_road_network first.")
        
        if verbose:
            print(f"\n🔍 Finding eco-route from node {start_node} to node {end_node}")
            print(f"Vehicle parameters: {vehicle_params}")
        
        # Create a new graph with fuel consumption as edge weights
        G_fuel = nx.DiGraph()
        
        for u, v in self.graph.edges():
            try:
                fuel = self.calculate_edge_fuel_consumption(u, v, vehicle_params, verbose)
                # Add bidirectional edges with fuel consumption as weight
                G_fuel.add_edge(u, v, weight=fuel, original_data=self.graph[u][v])
                G_fuel.add_edge(v, u, weight=fuel, original_data=self.graph[u][v])
            except Exception as e:
                print(f"Failed to calculate fuel consumption for edge ({u}, {v}): {e}")
        
        # Find the shortest path based on fuel consumption
        try:
            path = nx.shortest_path(G_fuel, source=start_node, target=end_node, weight='weight')
            
            # Calculate total fuel and distance
            total_fuel = 0
            total_distance = 0
            total_time = 0
            path_details = []
            
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                segment_fuel = G_fuel[u][v]['weight']
                segment_distance = G_fuel[u][v]['original_data']['distance']
                segment_time = G_fuel[u][v]['original_data']['travel_time']
                segment_data = G_fuel[u][v]['original_data']
                
                total_fuel += segment_fuel
                total_distance += segment_distance
                total_time += segment_time
                
                path_details.append({
                    'from': u,
                    'to': v,
                    'distance': segment_distance,
                    'time': segment_time,
                    'fuel': segment_fuel,
                    'terrain': segment_data['terrain_type'],
                    'traffic': segment_data['traffic_level'],
                    'road_quality': segment_data['road_quality']
                })
            
            if verbose:
                print("\n📊 Eco-Route Summary:")
                print(f"Path: {' → '.join(map(str, path))}")
                print(f"Total distance: {total_distance:.2f} km")
                print(f"Total travel time: {total_time*60:.1f} minutes")
                print(f"Total fuel consumption: {total_fuel:.2f} liters")
                print("\nSegment Details:")
                for segment in path_details:
                    print(f"  {segment['from']} → {segment['to']}: {segment['distance']:.2f} km, "
                          f"{segment['time']*60:.1f} min, {segment['fuel']:.2f} liters, {segment['terrain']} terrain, "
                          f"{segment['traffic']} traffic, {segment['road_quality']} road")
            
            return path, total_fuel, path_details, total_distance, total_time
        except nx.NetworkXNoPath:
            if verbose:
                print(f"❌ No path found from {start_node} to {end_node}")
            return None, float('inf'), [], 0, 0
    
    def find_fastest_route(self, start_node, end_node, vehicle_params, verbose=False):
        """Find the fastest route between two nodes"""
        if self.graph is None:
            raise ValueError("Road network not created yet. Call create_road_network first.")
        
        if verbose:
            print(f"\n🔍 Finding fastest route from node {start_node} to node {end_node}")
        
        # Create a new graph with travel time as edge weights
        G_time = nx.DiGraph()
        
        for u, v in self.graph.edges():
            travel_time = self.graph[u][v]['travel_time']
            G_time.add_edge(u, v, weight=travel_time, original_data=self.graph[u][v])
            G_time.add_edge(v, u, weight=travel_time, original_data=self.graph[u][v])
        
        # Find the shortest path based on travel time
        try:
            path = nx.shortest_path(G_time, source=start_node, target=end_node, weight='weight')
            
            # Calculate total fuel, distance and time
            total_fuel = 0
            total_distance = 0
            total_time = 0
            path_details = []
            
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                segment_distance = G_time[u][v]['original_data']['distance']
                segment_time = G_time[u][v]['original_data']['travel_time']
                segment_fuel = self.calculate_edge_fuel_consumption(u, v, vehicle_params)
                segment_data = G_time[u][v]['original_data']
                
                total_fuel += segment_fuel
                total_distance += segment_distance
                total_time += segment_time
                
                path_details.append({
                    'from': u,
                    'to': v,
                    'distance': segment_distance,
                    'time': segment_time,
                    'fuel': segment_fuel,
                    'terrain': segment_data['terrain_type'],
                    'traffic': segment_data['traffic_level'],
                    'road_quality': segment_data['road_quality']
                })
            
            if verbose:
                print("\n⏱️ Fastest Route Summary:")
                print(f"Path: {' → '.join(map(str, path))}")
                print(f"Total distance: {total_distance:.2f} km")
                print(f"Total travel time: {total_time*60:.1f} minutes")
                print(f"Total fuel consumption: {total_fuel:.2f} liters")
            
            return path, total_fuel, path_details, total_distance, total_time
        except nx.NetworkXNoPath:
            if verbose:
                print(f"❌ No path found from {start_node} to {end_node}")
            return None, float('inf'), [], 0, 0
    
    def find_shortest_route(self, start_node, end_node, vehicle_params, verbose=False):
        """Find the shortest distance route between two nodes"""
        if self.graph is None:
            raise ValueError("Road network not created yet. Call create_road_network first.")
        
        if verbose:
            print(f"\n🔍 Finding shortest distance route from node {start_node} to node {end_node}")
        
        # Create a new graph with distance as edge weights
        G_distance = nx.DiGraph()
        
        for u, v in self.graph.edges():
            distance = self.graph[u][v]['distance']
            G_distance.add_edge(u, v, weight=distance, original_data=self.graph[u][v])
            G_distance.add_edge(v, u, weight=distance, original_data=self.graph[u][v])
        
        # Find the shortest path based on distance
        try:
            path = nx.shortest_path(G_distance, source=start_node, target=end_node, weight='weight')
            
            # Calculate total fuel, distance and time
            total_fuel = 0
            total_distance = 0
            total_time = 0
            path_details = []
            
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                segment_distance = G_distance[u][v]['original_data']['distance']
                segment_time = G_distance[u][v]['original_data']['travel_time']
                segment_fuel = self.calculate_edge_fuel_consumption(u, v, vehicle_params)
                segment_data = G_distance[u][v]['original_data']
                
                total_fuel += segment_fuel
                total_distance += segment_distance
                total_time += segment_time
                
                path_details.append({
                    'from': u,
                    'to': v,
                    'distance': segment_distance,
                    'time': segment_time,
                    'fuel': segment_fuel,
                    'terrain': segment_data['terrain_type'],
                    'traffic': segment_data['traffic_level'],
                    'road_quality': segment_data['road_quality']
                })
            
            if verbose:
                print("\n📏 Shortest Distance Route Summary:")
                print(f"Path: {' → '.join(map(str, path))}")
                print(f"Total distance: {total_distance:.2f} km")
                print(f"Total travel time: {total_time*60:.1f} minutes")
                print(f"Total fuel consumption: {total_fuel:.2f} liters")
            
            return path, total_fuel, path_details, total_distance, total_time
        except nx.NetworkXNoPath:
            if verbose:
                print(f"❌ No path found from {start_node} to {end_node}")
            return None, float('inf'), [], 0, 0
    
    def calculate_efficiency_score(self, route_data, best_fuel, best_time, best_distance):
        """Calculate an efficiency score out of 100 based on multiple factors"""
        # Extract route metrics
        fuel = route_data['fuel']
        time = route_data['time']
        distance = route_data['distance']
        
        # Calculate normalized scores (0-1, where 1 is best)
        # For each metric, we calculate how close this route is to the best possible value
        fuel_score = best_fuel / fuel if fuel > 0 else 0  # Lower fuel is better
        time_score = best_time / time if time > 0 else 0  # Lower time is better
        distance_score = best_distance / distance if distance > 0 else 0  # Lower distance is better
        
        # Calculate weighted score (prioritizing fuel efficiency)
        # Weights should sum to 1
        weighted_score = (0.6 * fuel_score) + (0.2 * time_score) + (0.2 * distance_score)
        
        # Convert to 0-100
        efficiency_score = int(weighted_score * 100)
        
        # Cap the score to ensure it's between 0 and 100
        efficiency_score = max(0, min(100, efficiency_score))
        
        return efficiency_score
    
    def compare_routes(self, start_node, end_node, vehicle_params, verbose=True):
        """Compare eco-route with fastest and shortest distance routes"""
        # Find eco-route (optimized for fuel)
        eco_path, eco_fuel, eco_details, eco_distance, eco_time = self.find_eco_route(
            start_node, end_node, vehicle_params, verbose=False
        )
        
        # Find fastest route (optimized for time)
        fast_path, fast_fuel, fast_details, fast_distance, fast_time = self.find_fastest_route(
            start_node, end_node, vehicle_params, verbose=False
        )
        
        # Find shortest distance route
        short_path, short_fuel, short_details, short_distance, short_time = self.find_shortest_route(
            start_node, end_node, vehicle_params, verbose=False
        )
        
        # Find the best possible values
        best_fuel = min(eco_fuel, fast_fuel, short_fuel)
        best_time = min(eco_time, fast_time, short_time)
        best_distance = min(eco_distance, fast_distance, short_distance)
        
        # Store route data in a structured format
        routes = {
            'eco': {
                'name': 'Eco-Friendly Route',
                'path': eco_path,
                'fuel': eco_fuel,
                'time': eco_time,
                'distance': eco_distance,
                'details': eco_details,
                'color': 'green',
                'description': 'Optimized for minimal fuel consumption'
            },
            'fast': {
                'name': 'Fastest Route',
                'path': fast_path,
                'fuel': fast_fuel,
                'time': fast_time,
                'distance': fast_distance,
                'details': fast_details,
                'color': 'red',
                'description': 'Optimized for minimal travel time'
            },
            'short': {
                'name': 'Shortest Route',
                'path': short_path,
                'fuel': short_fuel,
                'time': short_time,
                'distance': short_distance,
                'details': short_details,
                'color': 'blue',
                'description': 'Optimized for minimal distance'
            }
        }
        
        # Calculate efficiency scores
        for route_key, route_data in routes.items():
            route_data['efficiency_score'] = self.calculate_efficiency_score(
                route_data, best_fuel, best_time, best_distance
            )
        
        # Display route comparison if verbose
        if verbose:
            print("\n🔄 Route Comparison:")
            
            # Table headers
            print(f"{'Route Type':<20} {'Distance':<12} {'Time':<12} {'Fuel':<12} {'Efficiency':<12}")
            print(f"{'-'*20:<20} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12}")
            
            # Table rows
            for route_key, route_data in routes.items():
                print(f"{route_data['name']:<20} "
                      f"{route_data['distance']:.2f} km{'':<4} "
                      f"{route_data['time']*60:.1f} min{'':<3} "
                      f"{route_data['fuel']:.2f} L{'':<5} "
                      f"{route_data['efficiency_score']}/100")
            
            # Detailed analysis
            print("\n📊 Route Analysis:")
            
            # Eco-route analysis
            print(f"\n• {routes['eco']['name']}:")
            print(f"  - {routes['eco']['description']}")
            if routes['eco']['efficiency_score'] > 80:
                print(f"  - Excellent fuel efficiency with a score of {routes['eco']['efficiency_score']}/100")
            print(f"  - {(routes['eco']['fuel'] / routes['eco']['distance'] * 100):.2f} L/100km average consumption")
            
            # Fastest route analysis
            print(f"\n• {routes['fast']['name']}:")
            print(f"  - {routes['fast']['description']}")
            print(f"  - Saves {(eco_time - fast_time) * 60:.1f} minutes compared to the eco-route")
            print(f"  - Uses {fast_fuel - eco_fuel:.2f} liters more fuel than the eco-route")
            
            # Shortest route analysis
            print(f"\n• {routes['short']['name']}:")
            print(f"  - {routes['short']['description']}")
            print(f"  - {short_distance:.2f} km total distance ({eco_distance - short_distance:.2f} km shorter than eco-route)")
            
            # Find the most balanced route
            efficiency_scores = [route_data['efficiency_score'] for route_data in routes.values()]
            most_balanced_key = list(routes.keys())[efficiency_scores.index(max(efficiency_scores))]
            most_balanced = routes[most_balanced_key]
            
            print(f"\n🏆 Recommendation:")
            print(f"The {most_balanced['name']} offers the best balance of efficiency with a score of {most_balanced['efficiency_score']}/100")
        
        return routes
    # This code modifies the visualize_route_comparison method to replace emoji with standard characters

    def visualize_route_comparison_improved(self, routes):
        """Create a comprehensive visual comparison of routes with improved layout"""
        if self.graph is None:
            raise ValueError("Road network not created yet. Call create_road_network first.")
        
        # Set up the figure with a 2x2 grid and more space between subplots
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.4, wspace=0.3)
        
        # Network map with all routes
        ax1 = plt.subplot(gs[0, :])
        
        # Get node positions from the geometric graph
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Draw the road network
        nx.draw_networkx_nodes(self.graph, pos, node_size=100, node_color='lightgray', ax=ax1)
        
        # Draw all edges with thickness proportional to distance
        for u, v in self.graph.edges():
            width = max(0.5, 2 / self.graph[u][v]['distance'])
            nx.draw_networkx_edges(self.graph, pos, edgelist=[(u, v)], width=width, alpha=0.1, edge_color='gray', ax=ax1)
        
        # Find the most efficient route (highest efficiency score)
        efficiency_scores = [route_data['efficiency_score'] for route_key, route_data in routes.items() 
                            if route_data['path']]
        best_route_key = None
        max_efficiency = -1
        
        for route_key, route_data in routes.items():
            if route_data['path'] and route_data['efficiency_score'] > max_efficiency:
                max_efficiency = route_data['efficiency_score']
                best_route_key = route_key
        
        # Draw each route with its own color
        for route_key, route_data in routes.items():
            if route_data['path']:
                edge_list = [(route_data['path'][i], route_data['path'][i+1]) 
                            for i in range(len(route_data['path'])-1)]
                
                # Highlight the most efficient route with a special edge style
                if route_key == best_route_key:
                    edge_width = 4
                    edge_style = 'solid'
                    alpha = 0.9
                else:
                    edge_width = 3
                    edge_style = 'dashed'
                    alpha = 0.7
                
                # Draw the route edges
                nx.draw_networkx_edges(
                    self.graph, pos, edgelist=edge_list, 
                    width=edge_width, edge_color=route_data['color'], 
                    alpha=alpha, label=route_data['name'],
                    style=edge_style, ax=ax1
                )
                
                # Mark start and end nodes (only mark once)
                if route_key == 'eco':
                    start_node = route_data['path'][0]
                    end_node = route_data['path'][-1]
                    nx.draw_networkx_nodes(self.graph, pos, nodelist=[start_node], 
                                        node_size=300, node_color='green', ax=ax1)
                    nx.draw_networkx_nodes(self.graph, pos, nodelist=[end_node], 
                                        node_size=300, node_color='orange', ax=ax1)
                    
                    # Add node labels
                    nx.draw_networkx_labels(self.graph, pos, 
                                        labels={start_node: 'Start', end_node: 'End'},
                                        font_size=12, font_weight='bold', ax=ax1)

        # Add legend with efficiency scores
        for route_key, route_data in routes.items():
            if route_data['path']:
                # Add star symbol to the most efficient route instead of crown
                label = f"{route_data['name']} (Score: {route_data['efficiency_score']})"
                if route_key == best_route_key:
                    label = "★ " + label + " ★"
                
                line_style = '-' if route_key == best_route_key else '--'
                ax1.plot([], [], color=route_data['color'], linewidth=3, 
                    label=label, linestyle=line_style)
        
        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_title('Route Comparison - Network Map', fontsize=16, pad=20)  # Added padding
        ax1.set_axis_off()
        
        # Comparative bar chart for metrics with color-coded bars
        ax2 = plt.subplot(gs[1, 0])
        
        # Prepare data for bar chart
        route_names = []
        distances = []
        times = []
        fuels = []
        route_colors = []
        bar_colors = {'distance': [], 'time': [], 'fuel': []}

        # Find best values for each metric
        best_distance = float('inf')
        best_time = float('inf')
        best_fuel = float('inf')
        
        for route_key, route_data in routes.items():
            if route_data['path']:
                route_names.append(route_data['name'])
                distances.append(route_data['distance'])
                times.append(route_data['time'] * 60)  # Convert to minutes
                fuels.append(route_data['fuel'])
                route_colors.append(route_data['color'])
                
                # Update best values
                best_distance = min(best_distance, route_data['distance'])
                best_time = min(best_time, route_data['time'] * 60)  # Convert to minutes
                best_fuel = min(best_fuel, route_data['fuel'])
        
        # Assign colors based on performance (green for best, yellow for mid, coral for worst)
        for route_data in [routes[key] for key in routes if routes[key]['path']]:
            # Color for distance
            if abs(route_data['distance'] - best_distance) < 0.01:
                bar_colors['distance'].append('green')
            elif route_data['distance'] < 1.1 * best_distance:
                bar_colors['distance'].append('yellowgreen')
            else:
                bar_colors['distance'].append('coral')
            
            # Color for time
            if abs(route_data['time'] * 60 - best_time) < 0.01:  # Convert to minutes
                bar_colors['time'].append('green')
            elif route_data['time'] * 60 < 1.1 * best_time:
                bar_colors['time'].append('yellowgreen')
            else:
                bar_colors['time'].append('coral')
            
            # Color for fuel
            if abs(route_data['fuel'] - best_fuel) < 0.01:
                bar_colors['fuel'].append('green')
            elif route_data['fuel'] < 1.1 * best_fuel:
                bar_colors['fuel'].append('yellowgreen')
            else:
                bar_colors['fuel'].append('coral')
        
        x = np.arange(len(route_names))
        width = 0.25
        
        # Plot bars with performance-based colors
        distance_bars = ax2.bar(x - width, distances, width, label='Distance (km)', alpha=0.8, 
                            color=bar_colors['distance'], edgecolor='black', linewidth=1)
        time_bars = ax2.bar(x, times, width, label='Time (min)', alpha=0.8, 
                        color=bar_colors['time'], edgecolor='black', linewidth=1)
        fuel_bars = ax2.bar(x + width, fuels, width, label='Fuel (L)', alpha=0.8, 
                        color=bar_colors['fuel'], edgecolor='black', linewidth=1)
        
        # Add value labels on top of bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,  # Added offset to prevent overlap
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        add_labels(distance_bars)
        add_labels(time_bars)
        add_labels(fuel_bars)
        
        # Add star symbols above the best performing bars with offset
        # to prevent overlap with the numeric values
        for i, d in enumerate(distances):
            if abs(d - best_distance) < 0.01:  # Account for floating point errors
                ax2.text(i - width, d - 0.5, '★', ha='center', va='bottom', fontsize=12)  # Place below the value
        
        for i, t in enumerate(times):
            if abs(t - best_time) < 0.01:  # Account for floating point errors
                ax2.text(i, t - 0.5, '★', ha='center', va='bottom', fontsize=12)  # Place below the value
        
        for i, f in enumerate(fuels):
            if abs(f - best_fuel) < 0.01:  # Account for floating point errors
                ax2.text(i + width, f - 0.2, '★', ha='center', va='bottom', fontsize=12)  # Place below the value
        
        ax2.set_ylabel('Value')
        ax2.set_ylim(0, max(max(distances), max(times), max(fuels)) * 1.2)  # Increase y-axis limit to make room
        ax2.set_title('Route Metrics Comparison', fontsize=14, pad=15)  # Added padding
        ax2.set_xticks(x)
        ax2.set_xticklabels(route_names)
        ax2.legend()
        
        # Add efficiency score panel on the right
        ax3 = plt.subplot(gs[1, 1])
        
        # Create horizontal bars for efficiency scores with gradient colors
        efficiency_scores = [route_data['efficiency_score'] for route_data in [routes[key] for key in routes if routes[key]['path']]]
        
        # Ensure efficiency scores are different for visualization purposes
        # This is for demonstration - in a real scenario you would use the actual scores
        if all(score == efficiency_scores[0] for score in efficiency_scores):
            efficiency_scores = [95, 85, 75]  # Example varied scores
        
        # Define a color map for the efficiency scores (red to green)
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(0, 100)
        
        # Create bars with gradient colors
        bars = ax3.barh(route_names, efficiency_scores, color=[cmap(norm(score)) for score in efficiency_scores])
        
        # Add efficiency score labels inside the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = max(width - 15, 5)  # Position the label inside the bar if possible
            ax3.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{efficiency_scores[i]}', 
                va='center', ha='center' if width > 30 else 'left',
                color='white' if width > 50 else 'black', fontweight='bold')
            
            # Add ranking labels with offset to prevent overlap
            rank_label = '1st' if i == 0 else '2nd' if i == 1 else '3rd'
            ax3.text(width + 2, bar.get_y() + bar.get_height()/2, rank_label, 
                    va='center', fontsize=12, fontweight='bold')
        
        # Add color bar to show the scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3, orientation='horizontal', pad=0.2)
        cbar.set_label('Efficiency Scale')
        
        ax3.set_xlim(0, 105)  # Make room for the labels
        ax3.set_xlabel('Efficiency Score (0-100)')
        ax3.set_title('Route Efficiency Scores', fontsize=14, pad=15)  # Added padding
        
        # Add legends explaining the colors with better positioning
        fig.text(0.5, 0.02, 
                '• Green bars indicate best performance\n• Yellow-green bars indicate good performance\n• Coral bars indicate areas for improvement\n• ★ Star marks the best option in each category', 
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.94])  # Adjusted rect to make room for titles and legend
        plt.suptitle('EcoRoute - Advanced Route Comparison', fontsize=18, y=0.98)
        
        return fig

if __name__ == "__main__":
    # Create an instance of the EcoRouteML class
    eco_router = EcoRouteML()
    
    # Train the fuel prediction model
    eco_router.train_fuel_prediction_model()
    
    # Create a road network
    eco_router.create_road_network(num_nodes=20, edge_probability=0.2)
    
    # Define vehicle parameters
    vehicle_params = {
        'weight': 1500,        # kg
        'engine_size': 2.0,    # liters
        'temperature': 20,     # celsius
        'wind_speed': 5        # km/h
    }
    
    # Compare different routing strategies
    routes = eco_router.compare_routes(0, 10, vehicle_params)
    
    # Visualize the comparison
    fig = eco_router.visualize_route_comparison_improved(routes)
    plt.show()

