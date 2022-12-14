import math


class EuclideanDistanceTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep count of the IDs
        # each time a new object id detected, count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Object boxes and ids
        object_bounding_box_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, width, height = rect
            cx = (x + x + width) // 2
            cy = (y + y + height) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                distance = math.hypot(cx - pt[0], cy - pt[1])
                # if object from frame t and frame t+1 are within distance they are treated as the same object
                if distance < 1:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    object_bounding_box_ids.append([x, y, width, height, id])
                    same_object_detected = True
                    break

            # If new object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                object_bounding_box_ids.append([x, y, width, height, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for object_bounding_box_id in object_bounding_box_ids:
            _, _, _, _, object_id = object_bounding_box_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return object_bounding_box_ids



