# GitHub Repository Security Setup

오픈소스 프로젝트의 PyPI 배포 workflow를 보호하기 위한 설정 가이드

## 🔒 필수 보안 설정

### 1. Environment Protection Rules
**Path**: `Settings` → `Environments` → `clyptq`

설정할 항목:
- ✅ **Required reviewers**: 최소 1명 (저장소 owner/admin)
- ✅ **Deployment branches**: Selected branches and tags only
  - Pattern: `refs/tags/v*` (v로 시작하는 태그만)
- ✅ **Wait timer**: 0 minutes (선택사항)

**효과**:
- 배포 전 승인 필수 (악의적 태그 push 차단)
- 특정 태그 패턴만 배포 허용

### 2. Tag Protection Rules (권장)
**Path**: `Settings` → `Code and automation` → `Tags` → `Protected tags`

설정:
- Pattern: `v*`
- ✅ **Allowed to create matching tags**: Restrict to administrators
- 또는 특정 팀/사용자만 허용

**효과**:
- 관리자만 `v*` 태그 생성 가능
- Contributor가 임의로 배포 태그 생성 불가

### 3. Actions General Settings
**Path**: `Settings` → `Actions` → `General`

**Fork pull request workflows**:
- ✅ **Run workflows from fork pull requests**: 체크 해제 (또는 "Require approval for first-time contributors")
- ✅ **Send secrets to workflows from fork pull requests**: 절대 체크 안함 ❌

**Workflow permissions**:
- ✅ **Read repository contents and packages permissions** (기본값)
- ❌ Write permissions 비활성화

**효과**:
- Fork에서의 PR은 secrets 접근 불가
- 악의적 contributor가 secrets 탈취 불가

### 4. Branch Protection Rules (선택사항)
**Path**: `Settings` → `Branches` → `Branch protection rules`

**Branch name pattern**: `master` (또는 `main`)

설정:
- ✅ **Require a pull request before merging**
- ✅ **Require status checks to pass before merging**
  - Required checks: `test (3.10)`, `test (3.11)`, `test (3.12)`
- ✅ **Require conversation resolution before merging**
- ✅ **Do not allow bypassing the above settings** (admins도 포함)

**효과**:
- 직접 push 방지
- CI 통과 필수

## 🛡️ 현재 적용된 Workflow 보호

`.github/workflows/publish.yml`의 보호 조건:

```yaml
publish:
  environment: clyptq  # 환경 승인 필수
  if: |
    github.repository == 'Clypt/clyptq' &&           # 원본 저장소만
    github.event_name == 'push' &&                   # Push 이벤트만
    startsWith(github.ref, 'refs/tags/v')            # v* 태그만
```

**차단되는 시나리오**:
- ❌ Fork 저장소에서 실행 (`github.repository` 체크)
- ❌ Pull request에서 실행 (`github.event_name` 체크)
- ❌ 브랜치 push에서 실행 (`startsWith(github.ref, 'refs/tags/v')` 체크)
- ❌ 환경 승인 없이 실행 (`environment: clyptq`)

## 📋 배포 체크리스트

배포 시 확인할 사항:

1. **코드 변경사항 검토**
   ```bash
   git log --oneline v0.2.3..HEAD
   git diff v0.2.3..HEAD
   ```

2. **로컬 테스트 통과 확인**
   ```bash
   pytest tests/ -v --cov=clyptq
   ```

3. **버전 업데이트**
   - `pyproject.toml`의 `version` 필드
   - `CLAUDE.md`의 버전 정보

4. **태그 생성 및 Push** (관리자만 가능)
   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

5. **GitHub Actions 모니터링**
   - Test job 통과 확인
   - Environment 승인 (required reviewers 설정 시)
   - Publish job 성공 확인

6. **PyPI 배포 검증**
   ```bash
   pip install clyptq==0.3.0
   python -c "import clyptq; print(clyptq.__version__)"
   ```

## 🚨 비상 대응

**악의적 배포 시도 발견 시**:

1. **즉시 조치**:
   - GitHub Actions workflow 취소
   - Environment에서 배포 거부
   - PyPI에서 패키지 yanked 처리

2. **사후 조치**:
   - PYPI_API_TOKEN 즉시 재생성
   - GitHub Secrets 업데이트
   - 보안 감사 로그 확인

3. **예방 조치**:
   - Tag protection rules 재확인
   - Environment reviewers 업데이트
   - 2FA 활성화 확인

## 📚 참고 자료

- [GitHub Actions Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [PyPI Security Best Practices](https://pypi.org/help/#apitoken)
- [Environment Protection Rules](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
