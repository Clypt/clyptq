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

### 2. Repository Rulesets (권장) - Tag Protection
**Path**: `Settings` → `Code and automation` → `Rules` → `Rulesets` → `New ruleset` → `New tag ruleset`

설정:
- **Ruleset Name**: "Protect release tags"
- **Enforcement status**: Active
- **Target tags**:
  - Include by pattern: `v*`
- **Rules**:
  - ✅ **Restrict creations**: Check
    - Restrict who can create matching tags
    - Add exception: Repository administrators (또는 특정 팀/역할만)
  - ✅ **Restrict deletions**: Check
  - ✅ **Restrict updates**: Check

**효과**:
- 관리자/지정된 사용자만 `v*` 태그 생성 가능
- Contributor가 임의로 배포 태그 생성/삭제 불가
- 기존 태그 덮어쓰기 방지

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

### 4. Repository Rulesets (선택사항) - Branch Protection
**Path**: `Settings` → `Code and automation` → `Rules` → `Rulesets` → `New ruleset` → `New branch ruleset`

**Branch name pattern**: `master` (또는 `main`)

설정:
- **Ruleset Name**: "Protect master branch"
- **Enforcement status**: Active
- **Target branches**:
  - Include by pattern: `master` (또는 `main`)
- **Rules**:
  - ✅ **Require a pull request before merging**
    - Required approving review count: 1
  - ✅ **Require status checks to pass**
    - Status checks that are required:
      - `test (3.10)`
      - `test (3.11)`
      - `test (3.12)`
  - ✅ **Require conversation resolution before merging**
  - ✅ **Block force pushes**
  - ✅ **Restrict deletions**

**Bypass list** (선택사항):
- Repository administrators (필요시 체크, 그렇지 않으면 비워두기)

**효과**:
- 직접 push 방지
- CI 통과 필수
- PR 리뷰 필수

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
- ❌ 환경 승인 없이 실행 (`environment: clyptq` + Required reviewers)
- ❌ Contributor의 임의 태그 생성 (Repository Rulesets - Tag protection)

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
   - Repository Rulesets 재확인 (Tag + Branch protection)
   - Environment reviewers 업데이트
   - 2FA 활성화 확인
   - Workflow 조건문 검증

## 📚 참고 자료

- [GitHub Actions Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [PyPI Security Best Practices](https://pypi.org/help/#apitoken)
- [Environment Protection Rules](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
- [Repository Rulesets (NEW)](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
- [Creating Tag Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
