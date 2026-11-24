export const routes = {
  home: '/', guide: '/guide', login: '/login', signup: '/signup', chat: '/chat', capture: '/capture', inventory: '/inventory',
  moveInfo: '/move-info', result: '/result', history: '/history',
} as const;

export type AppRoute = (typeof routes)[keyof typeof routes];

export function navigate(path: AppRoute) {
  window.history.pushState({}, '', path);
  window.dispatchEvent(new PopStateEvent('popstate'));
}
