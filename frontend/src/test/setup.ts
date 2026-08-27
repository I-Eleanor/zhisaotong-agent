// vitest 全局增强：jest-dom 匹配器（toBeInTheDocument 等）
import "@testing-library/jest-dom/vitest";

// jsdom 未实现 scrollIntoView，组件底部自动滚动在测试中直接予以跳过
if (typeof Element !== "undefined" && !Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}