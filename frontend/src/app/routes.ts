export const routes = {
  home: '/', guide: '/guide', login: '/login', signup: '/signup', chat: '/chat', capture: '/capture', inventory: '/inventory',
  moveInfo: '/move-info', analysis: '/analysis', result: '/result', history: '/history', historyDetail: '/history/detail',
  afterReview: '/after-review',
} as const;

export type AppRoute = (typeof routes)[keyof typeof routes];

export function navigate(path: AppRoute) {
  window.history.pushState({}, '', path);
  window.dispatchEvent(new PopStateEvent('popstate'));
}

export function openGuide() {
  window.history.pushState({ guideReturnTo: window.location.pathname + window.location.search }, '', routes.guide);
  window.dispatchEvent(new PopStateEvent('popstate'));
}

export function closeGuide() {
  if (window.history.state?.guideReturnTo) window.history.back();
  else navigate(routes.home);
}
