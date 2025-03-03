#include "inference.h"
#include "utils.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>  // 需要安装 nlohmann/json 库

int main(const int argc, const char **argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model_path> <image_path> [<output_path>]" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
	std::string output_path = "";
	if (argc > 3) {
		output_path = argv[3];
	}

    // 根据模型路径获取 metadata 文件路径，并读取 class_names
    const std::size_t pos = model_path.find_last_of("/");
    const std::string metadata_path = model_path.substr(0, pos + 1) + "metadata.yaml";
    const std::vector<std::string> class_names = GetClassNameFromMetadata(metadata_path);

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "ERROR: image is empty" << std::endl;
        return 1;
    }
	if (class_names.empty()) {
		std::cerr << "ERROR: class_names is empty" << std::endl;
		return 1;
	}

    const float confidence_threshold = 0.5;
    yolo::Inference inference(model_path, confidence_threshold);
    std::vector<yolo::Detection> detections = inference.RunInference(image);

    // 使用 nlohmann::json 构造 JSON 输出
    nlohmann::json json_output;
    json_output["detections"] = nlohmann::json::array();

    for (size_t i = 0; i < detections.size(); ++i) {
        const yolo::Detection &det = detections[i];
        nlohmann::json j;
        j["class_name"] = class_names[det.class_id];
        j["confidence"] = det.confidence;
        j["box"] = {
            {"x", det.box.x},
            {"y", det.box.y},
            {"width", det.box.width},
            {"height", det.box.height}
        };
        json_output["detections"].push_back(j);
    }
    std::cout << json_output.dump() << std::endl;

	if (!output_path.empty()) {
		DrawDetectedObject(image, detections, class_names);
		if (!cv::imwrite(output_path, image)) {
			std::cerr << "ERROR: Could not write image to " << output_path << std::endl;
			return 1;
		}
	}

    return 0;
}
