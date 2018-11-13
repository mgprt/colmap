#define TEST_NAME "feature/utils"
#include "util/testing.h"

#include "base/camera.h"
#include "base/database.h"
#include "feature/masking.h"

using namespace colmap;

const static std::string kMemoryDatabasePath = ":memory:";

BOOST_AUTO_TEST_SUITE(MaskReaderTestSuite)

BOOST_AUTO_TEST_CASE(ParseMaskList) {
  std::string mask_list = "1:dummymask1.jpg,2:another_dir/mask2.png";
  const auto mask_paths = FeatureMaskReader::ReadImageMaskString(mask_list);
  BOOST_CHECK_EQUAL(mask_paths.size(), 2);
  BOOST_CHECK(mask_paths.count(1) > 0);
  BOOST_CHECK(mask_paths.count(2) > 0);

  BOOST_CHECK_EQUAL(mask_paths.at(1), "dummymask1.jpg");
  BOOST_CHECK_EQUAL(mask_paths.at(2), "another_dir/mask2.png");
}

// TODO Can we test the file reading as well?

BOOST_AUTO_TEST_CASE(GetImageMasksFromCameraMasks) {
  Database database(kMemoryDatabasePath);
  Camera camera1;
  camera1.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera1.SetCameraId(database.WriteCamera(camera1));
  Camera camera2;
  camera2.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera2.SetCameraId(database.WriteCamera(camera2));

  Image image1;
  image1.SetName("image1");
  image1.SetCameraId(camera1.CameraId());
  image1.SetImageId(database.WriteImage(image1));

  Image image2;
  image2.SetName("image2");
  image2.SetCameraId(camera1.CameraId());
  image2.SetImageId(database.WriteImage(image2));

  Image image3;
  image3.SetName("image3");
  image3.SetCameraId(camera2.CameraId());
  image3.SetImageId(database.WriteImage(image3));

  std::unordered_map<camera_t, std::string> camera_masks;
  camera_masks[1] = "mask1";
  camera_masks[2] = "mask2";

  std::unordered_map<image_t, std::string> image_masks =
      FeatureMaskReader{database}.CameraMasksToImageMasks(camera_masks);

  BOOST_CHECK_EQUAL(image_masks.size(), 3);
  BOOST_CHECK_EQUAL(image_masks.at(1), "mask1");
  BOOST_CHECK_EQUAL(image_masks.at(2), "mask1");
  BOOST_CHECK_EQUAL(image_masks.at(3), "mask2");
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(FeatureMaskerTestSuite)

// TODO Can we test abortion if the dimensions do not match?

BOOST_AUTO_TEST_CASE(MaskImageFeatures) {
  Database database(kMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 200.0, 200, 200);
  camera.SetCameraId(database.WriteCamera(camera));
  Image image;
  image.SetImageId(1);
  image.SetName("image1");
  image.SetCameraId(camera.CameraId());
  database.WriteImage(image, true);

  auto check_num_features = [&](int reference_num) {
    BOOST_CHECK_EQUAL(database.NumKeypointsForImage(1), reference_num);
    BOOST_CHECK_EQUAL(database.NumDescriptorsForImage(1), reference_num);

    // Check that we didn't associate the wrong keypoints and descriptors
    const FeatureKeypoints keypoints = database.ReadKeypoints(1);
    const FeatureDescriptors descriptors = database.ReadDescriptors(1);
    BOOST_CHECK_EQUAL(keypoints.size(), descriptors.rows());
    const int num_keypoints = keypoints.size();
    for (int i = 0; i < num_keypoints; ++i) {
      const FeatureKeypoint kp = keypoints[i];
      BOOST_CHECK_EQUAL(kp.x, descriptors(i, 0));
      BOOST_CHECK_EQUAL(kp.y, descriptors(i, 127));
    }
  };

  auto setup_keypoints = [&]() {
    database.DeleteFeatures(1);
    FeatureKeypoints keypoints;
    keypoints.reserve(1521);
    FeatureDescriptors descriptors(1521, 128);

    for (int x = 5; x < 200; x += 5) {    // 39 cols
      for (int y = 5; y < 200; y += 5) {  // 39 rows
        size_t kp_idx = keypoints.size();
        descriptors.row(kp_idx) = Eigen::Matrix<uint8_t, 1, 128>::Ones();
        descriptors.block(kp_idx, 0, 1, 64) *= static_cast<uint8_t>(x);
        descriptors.block(kp_idx, 64, 1, 64) *= static_cast<uint8_t>(y);

        keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y));
      }
    }
    database.WriteKeypoints(1, keypoints);
    database.WriteDescriptors(1, descriptors);
    BOOST_CHECK_EQUAL(database.NumKeypointsForImage(1), 1521);
    check_num_features(1521);
  };

  auto create_rectangle_mask = [](int startrow, int startcol, int height,
                                  int width) {
    Bitmap mask;
    mask.Allocate(200, 200, false);
    mask.Fill(BitmapColor<uint8_t>(255));
    // There is no way to fill just a part of the image, so we just iterate over
    // all pixels in the area here
    for (int x = 0; x < width; ++x) {
      for (int y = 0; y < height; ++y) {
        mask.SetPixel(x + startcol, y + startrow, BitmapColor<uint8_t>(0));
      }
    }
    return mask;
  };

  // Mask1 (all white - nothing masked out)
  Bitmap mask1 = create_rectangle_mask(0, 0, 0, 0);
  DbImageFeatureMasker feature_masker(database);
  setup_keypoints();
  feature_masker.MaskImageFeatures({{1, std::make_shared<Bitmap>(mask1)}});
  check_num_features(1521);

  // Mask2 - mask out features on the right side of the image.
  Bitmap mask2 = create_rectangle_mask(0, 100, 200, 100);
  setup_keypoints();
  feature_masker.MaskImageFeatures({{1, std::make_shared<Bitmap>(mask2)}});
  check_num_features(741);
  // Check the position of the remaining keypoints
  {
    const FeatureKeypoints keypoints = database.ReadKeypoints(1);
    for (const FeatureKeypoint& kp : keypoints) {
      BOOST_CHECK_LT(kp.x, 100);
    }
  }

  // Mask3 - mask out area in the center
  Bitmap mask3 = create_rectangle_mask(50, 50, 101, 101);
  mask3.Write("mask3.png");
  setup_keypoints();
  feature_masker.MaskImageFeatures({{1, std::make_shared<Bitmap>(mask3)}});
  check_num_features(1080);  // 2*9*39 + 2*21*9
  {
    const FeatureKeypoints keypoints = database.ReadKeypoints(1);
    for (const FeatureKeypoint& kp : keypoints) {
      BOOST_CHECK(kp.x < 50 || kp.x > 150 || kp.y < 50 || kp.y > 150);
    }
  }

  // Mask4 - everything masked out
  Bitmap mask4 = create_rectangle_mask(0, 0, 200, 200);
  setup_keypoints();
  feature_masker.MaskImageFeatures({{1, std::make_shared<Bitmap>(mask4)}});
  check_num_features(0);
}

BOOST_AUTO_TEST_SUITE_END()