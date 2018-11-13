#include "masking.h"

#include <vector>

#include "base/database.h"
#include "util/misc.h"

namespace colmap {

// anonymous namespace to hide FeatureMaskReader functions outside this file
namespace {

template <typename id_type>
std::unordered_map<id_type, std::string> ParseMaskPaths(
    const std::vector<std::string>& mask_paths) {
  std::unordered_map<id_type, std::string> mask_path_map;
  for (const auto& mask_path : mask_paths) {
    const auto pair = StringSplit(mask_path, ":");
    CHECK(pair.size() == 2);
    const id_type id = std::stoul(pair[0]);
    const std::string& path = pair[1];
    mask_path_map.emplace(id, path);
  }

  return mask_path_map;
}

template <typename id_type>
std::unordered_map<id_type, std::string> ReadMaskString(
    const std::string& mask_string) {
  std::vector<std::string> mask_paths = CSVToVector<std::string>(mask_string);

  return ParseMaskPaths<id_type>(mask_paths);
}

template <typename id_type>
std::unordered_map<id_type, std::string> ReadMaskFile(const std::string& path) {
  std::vector<std::string> mask_paths = ReadTextFileLines(path);

  return ParseMaskPaths<id_type>(mask_paths);
}
}  // namespace

FeatureMaskReader::FeatureMaskReader(const colmap::Database& database)
    : database_(database) {}

std::unordered_map<image_t, std::string> FeatureMaskReader::ReadImageMaskString(
    const std::string& mask_string) {
  return ReadMaskString<image_t>(mask_string);
}

std::unordered_map<image_t, std::string> FeatureMaskReader::ReadImageMaskFile(
    const std::string& path) {
  return ReadMaskFile<image_t>(path);
}

std::unordered_map<camera_t, std::string>
FeatureMaskReader::ReadCameraMaskString(const std::string& mask_string) {
  return ReadMaskString<camera_t>(mask_string);
}

std::unordered_map<camera_t, std::string> FeatureMaskReader::ReadCameraMaskFile(
    const std::string& path) {
  return ReadMaskFile<camera_t>(path);
}

std::unordered_map<image_t, std::string>
FeatureMaskReader::CameraMasksToImageMasks(
    const std::unordered_map<colmap::camera_t, std::string>& camera_masks)
    const {
  std::unordered_map<image_t, std::string> image_masks;
  for (const auto& camera_mask : camera_masks) {
    std::vector<Image> camera_images =
        database_.ReadCameraImages(camera_mask.first);
    const std::string& mask_path = camera_mask.second;
    for (const Image& image : camera_images) {
      image_masks.emplace(image.ImageId(), mask_path);
    }
  }
  return image_masks;
}

std::unordered_map<image_t, std::shared_ptr<Bitmap>>
FeatureMaskReader::ReadMasks(
    const std::unordered_map<colmap::image_t, std::string> image_masks) {
  std::unordered_map<image_t, std::shared_ptr<Bitmap>> masks;

  // Keep track of the masks that we already read, so that we can just add a
  // pointer if we refer to the same one twice. It's generally inefficient to
  // copy shared pointers, but it should be negligible compared to the mask
  // reading and makes the handling later much easier.
  std::unordered_map<std::string, std::shared_ptr<Bitmap>> read_masks;
  for (const auto& image_mask : image_masks) {
    const std::string& path = image_mask.second;
    if (read_masks.count(path) == 0) {
      CHECK(ExistsFile(path));
      Bitmap mask;
      CHECK(mask.Read(path, false));
      read_masks.emplace(path, std::make_shared<Bitmap>(mask));
    }

    masks.emplace(image_mask.first, read_masks.at(path));
  }
  return masks;
}

// anonymous namespace to hide DbImageFeatureMasker functions in other files.
namespace {

// Simple version that only considers the feature position. It would make sense
// to also use the feature scale to check the whole feature area.
bool IsMaskedOut(const Bitmap& mask, const float x, const float y) {
  // If any neighboring pixel is masked out, set the position as masked out
  const std::vector<int> x_coords{
      {static_cast<int>(floor(x)), static_cast<int>(ceil(x))}};
  const std::vector<int> y_coords{
      {static_cast<int>(floor(y)), static_cast<int>(ceil(y))}};

  colmap::BitmapColor<uint8_t> pixel_color{255};
  uint8_t mask_color{0};

  for (int x_coord : x_coords) {
    if (x_coord < 0 || x_coord >= mask.Width()) {
      continue;
    }
    for (int y_coord : y_coords) {
      if (y_coord < 0 || y_coord >= mask.Height()) {
        continue;
      }
      CHECK(mask.GetPixel(x_coord, y_coord, &pixel_color));
      if (pixel_color.r == mask_color) {
        return true;
      }
    }
  }
  return false;
}

std::vector<size_t> GetRemainingFeatureIndices(
    const Bitmap& mask, const FeatureKeypoints& keypoints) {
  std::vector<size_t> remaining_indices;
  remaining_indices.reserve(keypoints.size());

  const size_t num_keypoints = keypoints.size();
  for (size_t i = 0; i < num_keypoints; ++i) {
    const FeatureKeypoint& kp = keypoints[i];
    if (!IsMaskedOut(mask, kp.x, kp.y)) {
      remaining_indices.push_back(i);
    }
  }

  return remaining_indices;
}

}  // namespace

DbImageFeatureMasker::DbImageFeatureMasker(const colmap::Database& database)
    : database_(database) {}

void DbImageFeatureMasker::MaskImageFeatures(
    const std::unordered_map<colmap::image_t, std::shared_ptr<Bitmap>>&
        mask_files) const {
  for (const auto& image_mask : mask_files) {
    const colmap::image_t image_id = image_mask.first;
    const Bitmap& mask = *image_mask.second;

    std::cout << "Mask features in image " << image_id << '\n';

    CHECK(database_.ExistsImage(image_id));

    // Make sure that image and mask have the same size, otherwise something is
    // wrong here.
    std::pair<size_t, size_t> image_size = database_.ReadImageSize(image_id);

    CHECK_EQ(image_size.first, mask.Width());
    CHECK_EQ(image_size.second, mask.Height());

    // All features are stored as a blob in the database, so we can only read
    // and write them together
    const FeatureKeypoints keypoints = database_.ReadKeypoints(image_id);

    std::vector<size_t> remaining_feature_indices =
        GetRemainingFeatureIndices(mask, keypoints);
    const size_t num_remaining_features = remaining_feature_indices.size();

    // We can leave the database features untouched if nothing changed
    if (num_remaining_features < keypoints.size()) {
      FeatureKeypoints remaining_keypoints;
      remaining_keypoints.reserve(remaining_feature_indices.size());

      for (const size_t keypoint_index : remaining_feature_indices) {
        remaining_keypoints.push_back(keypoints.at(keypoint_index));
      }

      // We also need to update the feature descriptors in the database or we
      // will end up in an inconsistent state. Do this in a separate loop to
      // hopefully get better cache usage.
      const FeatureDescriptors feature_descriptors =
          database_.ReadDescriptors(image_id);
      CHECK_EQ(feature_descriptors.rows(), keypoints.size());
      FeatureDescriptors remaining_descriptors{remaining_feature_indices.size(),
                                               feature_descriptors.cols()};
      for (size_t i = 0; i < num_remaining_features; ++i) {
        const size_t old_feature_index = remaining_feature_indices.at(i);
        remaining_descriptors.row(i) =
            feature_descriptors.row(old_feature_index);
      }

      // Delete the old keypoints / descriptors in the database and write the
      // new ones.
      database_.DeleteFeatures(image_id);
      database_.WriteKeypoints(image_id, remaining_keypoints);
      database_.WriteDescriptors(image_id, remaining_descriptors);
    }
  }
}

};  // namespace colmap