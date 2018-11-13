#ifndef COLMAP_SRC_FEATURE_MASKING_H_
#define COLMAP_SRC_FEATURE_MASKING_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "util/bitmap.h"
#include "util/types.h"

namespace colmap {

class Database;

class FeatureMaskReader {
 public:
  FeatureMaskReader(const Database& database);

  static std::unordered_map<image_t, std::string> ReadImageMaskString(
      const std::string& mask_string);
  static std::unordered_map<image_t, std::string> ReadImageMaskFile(
      const std::string& path);
  static std::unordered_map<image_t, std::string> ReadCameraMaskString(
      const std::string& mask_string);
  static std::unordered_map<image_t, std::string> ReadCameraMaskFile(
      const std::string& path);

  static std::unordered_map<image_t, std::shared_ptr<Bitmap>> ReadMasks(
      const std::unordered_map<image_t, std::string> image_masks);

  std::unordered_map<image_t, std::string> CameraMasksToImageMasks(
      const std::unordered_map<camera_t, std::string>& camera_masks) const;

 private:
  const Database& database_;
};

class DbImageFeatureMasker {
 public:
  DbImageFeatureMasker(const Database& database);

  void MaskImageFeatures(
      const std::unordered_map<colmap::image_t, std::shared_ptr<Bitmap>>&
          mask_files) const;

 private:
  const Database& database_;
};
}  // namespace colmap

#endif  // COLMAP_SRC_FEATURE_MASKING_H_