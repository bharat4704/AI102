"""
============================================================
AI-102 | Program 15 — Face Analysis
Service : Azure AI Face
Skill   : Analyze images with Azure AI Vision (Face)
============================================================
Features:
  • Face detection with attributes
  • Face verification (same person?)
  • Face identification (who is this?)
  • Liveness detection concepts
  • Responsible AI note: attributes limited post-June 2023
============================================================
IMPORTANT: Emotion, gender, age attributes require
  Limited Access approval from Microsoft.
  Standard detection returns: blur, exposure, noise,
  mask, occlusion, head pose, quality.
============================================================
"""

import os
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeTypeDetection01,
    FaceAttributeTypeDetection03,
    FaceAttributeTypeRecognition04,
    QualityForRecognition,
)
from azure.core.credentials import AzureKeyCredential

FACE_ENDPOINT = os.getenv("AZURE_FACE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
FACE_KEY      = os.getenv("AZURE_FACE_KEY", "<your-face-key>")

def get_client():
    return FaceClient(
        endpoint=FACE_ENDPOINT,
        credential=AzureKeyCredential(FACE_KEY)
    )

# ── 1. Face Detection ─────────────────────────────────────
def detect_faces(image_url: str) -> list:
    """
    Detect faces in an image and return attributes.
    Returns face ID (temporary, 24hr expiry) and attributes.
    """
    client = get_client()

    faces = client.detect_from_url(
        url=image_url,
        detection_model=FaceDetectionModel.DETECTION03,       # Latest
        recognition_model=FaceRecognitionModel.RECOGNITION04, # Latest
        return_face_id=True,
        return_face_attributes=[
            FaceAttributeTypeDetection03.HEAD_POSE,
            FaceAttributeTypeDetection03.MASK,
            FaceAttributeTypeDetection03.BLUR,
            FaceAttributeTypeDetection03.EXPOSURE,
            FaceAttributeTypeDetection03.NOISE,
            FaceAttributeTypeDetection03.QUALITY_FOR_RECOGNITION,
        ],
        return_face_landmarks=True,
    )

    print("\n" + "="*65)
    print("  FACE DETECTION")
    print("="*65)
    print(f"  Image URL: {image_url[:70]}")
    print(f"  Faces detected: {len(faces)}")

    for i, face in enumerate(faces):
        rect = face.face_rectangle
        print(f"\n  Face {i+1}:")
        print(f"    Face ID  : {face.face_id}")
        print(f"    Location : top={rect.top}, left={rect.left}, "
              f"width={rect.width}, height={rect.height}")

        if face.face_attributes:
            attr = face.face_attributes
            if attr.head_pose:
                print(f"    Head Pose: pitch={attr.head_pose.pitch:.1f}° "
                      f"roll={attr.head_pose.roll:.1f}° "
                      f"yaw={attr.head_pose.yaw:.1f}°")
            if attr.mask:
                print(f"    Mask     : type={attr.mask.type}, "
                      f"nose/mouth covered={attr.mask.nose_and_mouth_covered}")
            if attr.blur:
                print(f"    Blur     : level={attr.blur.blur_level}, "
                      f"value={attr.blur.value:.2f}")
            if attr.quality_for_recognition:
                print(f"    Quality  : {attr.quality_for_recognition}")

        if face.face_landmarks:
            lm = face.face_landmarks
            print(f"    Landmarks: pupil_left=({lm.pupil_left.x:.0f},{lm.pupil_left.y:.0f}) "
                  f"pupil_right=({lm.pupil_right.x:.0f},{lm.pupil_right.y:.0f})")

    return [face.face_id for face in faces if face.face_id]

# ── 2. Face Verification ──────────────────────────────────
def verify_faces(face_id_1: str, face_id_2: str) -> None:
    """
    Verify if two face IDs belong to the same person.
    Returns: is_identical (bool) + confidence score.
    """
    client = get_client()

    result = client.verify_face_to_face(
        face_id1=face_id_1,
        face_id2=face_id_2
    )

    print("\n" + "="*65)
    print("  FACE VERIFICATION")
    print("="*65)
    print(f"  Face 1 ID  : {face_id_1}")
    print(f"  Face 2 ID  : {face_id_2}")
    print(f"  Identical  : {result.is_identical}")
    print(f"  Confidence : {result.confidence:.4f}")

# ── 3. Person Group — Identify Faces ─────────────────────
def create_and_train_person_group(group_id: str, person_data: dict) -> None:
    """
    Create a person group, add faces, train the model.
    person_data = {'person_name': ['image_url_1', 'image_url_2']}
    """
    client = get_client()

    # Create person group
    print(f"\n  Creating person group: {group_id}")
    client.person_group.create(
        person_group_id=group_id,
        name=group_id,
        recognition_model=FaceRecognitionModel.RECOGNITION04
    )

    person_ids = {}
    for name, urls in person_data.items():
        # Add person
        person = client.person_group.person.create(
            person_group_id=group_id,
            name=name
        )
        person_ids[name] = person.person_id
        print(f"  Added person: {name} (ID: {person.person_id})")

        # Add face images for this person
        for url in urls:
            try:
                client.person_group.person.add_face_from_url(
                    person_group_id=group_id,
                    person_id=person.person_id,
                    url=url
                )
                print(f"    ✅ Face added from: {url[:50]}")
            except Exception as e:
                print(f"    ❌ Failed: {e}")

    # Train the group
    print(f"\n  Training person group...")
    poller = client.person_group.train(person_group_id=group_id)

    # Wait for training
    import time
    while True:
        status = client.person_group.get_training_status(group_id)
        print(f"  Training status: {status.status}")
        if status.status.value in ["succeeded", "failed"]:
            break
        time.sleep(2)

    print(f"  Training complete. Person IDs: {person_ids}")
    return person_ids

def identify_faces(group_id: str, image_url: str) -> None:
    """
    Identify detected faces against a trained person group.
    """
    client = get_client()

    # First detect faces in the query image
    faces = client.detect_from_url(
        url=image_url,
        detection_model=FaceDetectionModel.DETECTION03,
        recognition_model=FaceRecognitionModel.RECOGNITION04,
        return_face_id=True
    )

    face_ids = [f.face_id for f in faces if f.face_id]
    if not face_ids:
        print("  No faces detected in query image.")
        return

    # Identify
    results = client.identify_from_person_group(
        face_ids=face_ids,
        person_group_id=group_id,
        max_num_of_candidates_returned=3,
        confidence_threshold=0.5
    )

    print("\n" + "="*65)
    print("  FACE IDENTIFICATION")
    print("="*65)
    print(f"  Query image: {image_url[:70]}")
    print(f"  Faces found: {len(faces)}")

    for identify_result in results:
        print(f"\n  Face ID: {identify_result.face_id}")
        if identify_result.candidates:
            for candidate in identify_result.candidates:
                print(f"    Candidate: person_id={candidate.person_id} "
                      f"confidence={candidate.confidence:.4f}")
        else:
            print("    No match found in group.")

# ── 4. Find Similar Faces ─────────────────────────────────
def find_similar_faces(query_face_id: str, candidate_face_ids: list[str]) -> None:
    """
    Find faces most similar to the query face.
    Does NOT require trained model — just face IDs.
    """
    client = get_client()

    results = client.find_similar(
        face_id=query_face_id,
        face_ids=candidate_face_ids,
        max_num_of_candidates_returned=5,
        mode="matchPerson"     # 'matchPerson' or 'matchFace'
    )

    print("\n" + "="*65)
    print("  FIND SIMILAR FACES")
    print("="*65)
    print(f"  Query face: {query_face_id}")
    print(f"  Candidates: {len(candidate_face_ids)}")
    print(f"  Matches found: {len(results)}")

    for result in results:
        print(f"    Face ID: {result.face_id} | Confidence: {result.confidence:.4f}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    test_url = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"

    # Detect
    face_ids = detect_faces(test_url)

    # If 2+ faces detected, verify pair
    if len(face_ids) >= 2:
        verify_faces(face_ids[0], face_ids[1])
        find_similar_faces(face_ids[0], face_ids[1:])

    print("\n  KEY POINTS FOR AI-102:")
    print("  • FaceClient uses FACE endpoint + key (different from Vision)")
    print("  • detection_model: DETECTION01, 02, 03 (use 03 — latest)")
    print("  • recognition_model: RECOGNITION01-04 (use 04 — latest)")
    print("  • Face IDs are TEMPORARY — expire after 24 hours")
    print("  • Person Groups persist — used for identification")
    print("  • verify_face_to_face() = same person check (1:1)")
    print("  • identify_from_person_group() = who is this? (1:N)")
    print("  • find_similar() = similar faces without training (1:N)")
    print("  • Emotion/age/gender attributes = Limited Access only")
    print("  • Liveness detection prevents photo spoofing attacks")
    print("="*65 + "\n")
