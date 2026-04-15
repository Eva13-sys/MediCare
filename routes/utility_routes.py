from flask import Blueprint, request, jsonify

util_bp = Blueprint("utils", __name__)

def init_util_routes(chatModel, summarize, validate, get_advice):

    @util_bp.route("/summarize", methods=["POST"])
    def summarize_route():
        text = request.json.get("text")
        return jsonify({"summary": summarize(chatModel, text)})

    @util_bp.route("/validate", methods=["POST"])
    def validate_route():
        answer = request.json.get("answer")
        return jsonify(validate(answer))

    @util_bp.route("/advice", methods=["POST"])
    def advice():
        symptoms = request.json.get("symptoms")
        return jsonify({"advice": get_advice(chatModel, symptoms)})

    return util_bp