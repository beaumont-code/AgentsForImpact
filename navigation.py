"""
Navigation module: gets walking directions using Apple MapKit via PyObjC.
Falls back to a simplified text-based response if MapKit is unavailable.
"""
import threading


def get_directions(destination: str, origin: str = None) -> str:
    """
    Tool handler: get walking directions from origin to destination.

    Args:
        destination: Place name or address (e.g., "Starbucks on Market St")
        origin: Starting location (optional, e.g., "500 Howard St, San Francisco")

    Returns:
        Formatted string with step-by-step walking directions.
    """
    try:
        return _get_directions_mapkit(destination, origin)
    except Exception as e:
        print(f"MapKit failed ({e}), using fallback")
        return _get_directions_fallback(destination, origin)


def _get_directions_mapkit(destination: str, origin: str = None) -> str:
    """Get walking directions using Apple MapKit via PyObjC."""
    import MapKit
    import CoreLocation

    result = {"steps": None, "error": None}
    event = threading.Event()

    def _do_geocode_and_directions():
        geocoder = CoreLocation.CLGeocoder.alloc().init()

        def on_dest_geocoded(placemarks, error):
            if error or not placemarks:
                result["error"] = f"Could not find location: {destination}"
                event.set()
                return

            dest_placemark = placemarks[0]
            dest_mapitem = MapKit.MKMapItem.alloc().initWithPlacemark_(
                MapKit.MKPlacemark.alloc().initWithPlacemark_(dest_placemark)
            )

            if origin:
                def on_origin_geocoded(origin_pms, origin_err):
                    if origin_err or not origin_pms:
                        result["error"] = f"Could not find origin: {origin}"
                        event.set()
                        return
                    origin_placemark = origin_pms[0]
                    origin_mapitem = MapKit.MKMapItem.alloc().initWithPlacemark_(
                        MapKit.MKPlacemark.alloc().initWithPlacemark_(origin_placemark)
                    )
                    _calculate_directions(origin_mapitem, dest_mapitem)

                geocoder.geocodeAddressString_completionHandler_(origin, on_origin_geocoded)
            else:
                origin_mapitem = MapKit.MKMapItem.mapItemForCurrentLocation()
                _calculate_directions(origin_mapitem, dest_mapitem)

        def _calculate_directions(source_item, dest_item):
            request = MapKit.MKDirections.Request.alloc().init()
            request.setSource_(source_item)
            request.setDestination_(dest_item)
            request.setTransportType_(1)  # MKDirectionsTransportTypeWalking

            directions = MapKit.MKDirections.alloc().initWithRequest_(request)

            def on_directions_calculated(response, error):
                if error:
                    result["error"] = f"Could not calculate directions: {error}"
                    event.set()
                    return

                routes = response.routes()
                if not routes:
                    result["error"] = "No walking route found to that destination."
                    event.set()
                    return
                route = routes[0]
                steps = []
                for step in route.steps():
                    instruction = step.instructions()
                    distance = step.distance()
                    if instruction:
                        if distance > 0:
                            feet = int(distance * 3.281)
                            steps.append(f"{instruction} for {feet} feet")
                        else:
                            steps.append(instruction)

                total_distance = int(route.distance() * 3.281)
                total_time = int(route.expectedTravelTime() / 60)

                result["steps"] = {
                    "steps": steps,
                    "total_distance_feet": total_distance,
                    "total_time_minutes": total_time,
                }
                event.set()

            directions.calculateDirectionsWithCompletionHandler_(on_directions_calculated)

        geocoder.geocodeAddressString_completionHandler_(destination, on_dest_geocoded)

    _do_geocode_and_directions()

    event.wait(timeout=15)

    if result["error"]:
        return f"Navigation error: {result['error']}"

    if result["steps"] is None:
        return "Could not get directions. The request timed out."

    steps_data = result["steps"]
    lines = [f"Walking directions to {destination}:"]
    lines.append(f"Total distance: about {steps_data['total_distance_feet']} feet")
    lines.append(f"Estimated time: about {steps_data['total_time_minutes']} minutes")
    lines.append("")
    for i, step in enumerate(steps_data["steps"], 1):
        lines.append(f"Step {i}: {step}")

    return "\n".join(lines)


def _get_directions_fallback(destination: str, origin: str = None) -> str:
    """Fallback when MapKit is unavailable."""
    origin_text = f" from {origin}" if origin else ""
    return (
        f"I'd like to help you get to {destination}{origin_text}, but I'm having "
        f"trouble accessing the maps service right now. Could you ask someone nearby "
        f"for directions, or try describing landmarks around you so I can help orient you?"
    )
