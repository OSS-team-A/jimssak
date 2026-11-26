export const routes = {
  home: '/', guide: '/guide', login: '/login', signup: '/signup', chat: '/chat', capture: '/capture', inventory: '/inventory',
  moveInfo: '/move-info', analysis: '/analysis', result: '/result', history: '/history',
  postmove: '/postmove', comparison: '/comparison',
} as const;

export type AppRoute = (typeof routes)[keyof typeof routes];

export function navigate(path: AppRoute) {
  window.history.pushState({}, '', path);
  window.dispatchEvent(new PopStateEvent('popstate'));
}
