class DecisionModule:
    """
    High-level decision making based on perception data.
    Adjusts target speed and handles traffic rules.
    """
    def __init__(self):
        self.base_speed = 30.0  # Default target speed in km/h
        self.current_target_speed = self.base_speed
        self.emergency_stop = False

    def update(self, perception_data):
        """
        Updates driving decisions based on detected objects and lanes.
        """
        objects = perception_data.get('objects', [])
        
        self.emergency_stop = False
        new_target_speed = self.base_speed

        for obj in objects:
            label = obj.get('label')
            confidence = obj.get('confidence', 0.0)
            
            if confidence < 0.4:
                continue

            # 1. Traffic Light Handling
            if label == 'traffic_light_red':
                self.emergency_stop = True
                new_target_speed = 0.0
            elif label == 'traffic_light_green':
                self.emergency_stop = False
                new_target_speed = self.base_speed

            # 2. Speed Limit Handling
            speed_map = {
                'speed_sign_30': 30.0,
                'speed_sign_60': 60.0,
                'speed_sign_90': 90.0
            }
            if label in speed_map:
                self.base_speed = speed_map[label]
                new_target_speed = self.base_speed

            # 3. Obstacle Avoidance (Simplified)
            # If a vehicle, pedestrian, or bike is too close (distance should be provided by perception)
            if label in ['vehicle', 'pedestrian', 'bike']:
                distance = obj.get('distance', 100.0)
                if distance < 10.0:  # Within 10 meters
                    self.emergency_stop = True
                    new_target_speed = 0.0

        self.current_target_speed = new_target_speed

    def get_target_speed(self):
        """Returns the calculated target speed in km/h."""
        return self.current_target_speed

    def is_stop_required(self):
        """Returns True if the vehicle should come to a complete stop."""
        return self.emergency_stop
